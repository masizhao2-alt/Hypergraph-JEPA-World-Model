import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import numpy as np
from functools import partial

from .pos_embedding import MaeEmbeddings
from .common import RMSNorm, LayerNorm, IdentityBlock, cosine_similarity_loss, _USE_APEX
from .combiners import ExpertCombiner, MeanCombiner, MaxCombiner, ImageCombiner, DualCombiner
from .layers import (
    MHHESA, 
    MHEAgg, 
    MHEDist, 
    FFN_GEGLU, 
    Split_FFN_GEGLU,
    PatchExtractor, 
    ViGStem,
    _ATTN_PRECISION,
    _USE_XFORMERS,
    _NORMALIZE_QK,
)

# Projection network for generating adj features when x=xadj
# - Reverts to Identity when split_mode=True
class AdjProjector(nn.Module):
    def __init__(self, dim, dim_adj, split_mode=True):
        super(AdjProjector, self).__init__()
        if not split_mode:
            print(" :: Initializing AdjProjector :: ")
            
        self.split_mode = split_mode
        self.proj = None if split_mode else nn.Linear(dim, dim_adj, bias=False)

    def forward(self, xfeat, xadj):
        if self.split_mode:
            return xadj # identity
        else:
            return self.proj(xfeat)

# a helper layer to test the different paths
class AssertBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super(AssertBlock, self).__init__()
        print(" :: Initializing AssertBlock :: ")
        
    def forward(self, *args, **kwargs):
        assert False, "This block should never be called!"

#_AssertBlock = AssertBlock()
_AssertBlock = None
        
class HyperGraphBlock(nn.Module):
    def __init__(self, dim, 
                 num_heads, 
                 adj_dim=None,
                 head_dim=-1,
                 bind_qkv=False, 
                 split_mode=True, 
                 joint_mlp=False,
                 drop_path=0.0,
                 pop_level=None,
                 skip_last=False,
                 drop_adjff=False,
                 use_xformers=True,
                 correct_mod=False,
                 use_nodesa=True,
                 shared_adjproj=False,
                ):
        super().__init__()

        print(f"Making HyperGraphBlock, skip_last={skip_last}, drop_adjff={drop_adjff}")

        # this will use the mode dim for adjacency and content independently
        self.split_mode = int(split_mode)
        self.dim = dim
        self.pop_level = pop_level
        self.skip_last = skip_last
        self.drop_adjff = drop_adjff
        self.use_nodesa = use_nodesa
        self.joint_mlp = joint_mlp
        self.shared_adjproj = shared_adjproj
        self._SAVESTATE = False

        # in the last block if skip_last is set, then
        # - the vertex FFN will not be used
        # - the edge dist attn will not be used
        
        if head_dim == -1:
            head_dim = dim // num_heads
        
        # follow a standard transformer block
        # MHSA, which performs self-attention over the grouped nodes
        if use_nodesa:
            self.node_sa = MHHESA(dim, num_heads, head_dim, bind_qkv=bind_qkv, drop_path=drop_path, use_xformers=use_xformers)
        else:
            self.node_sa = _AssertBlock

        # MHCA, which performs aggregation from nodes -> edges, and then performs 
        #       distribution from edges -> nodes
        self.edge_agg = MHEAgg(dim, num_heads, head_dim, bind_qkv=bind_qkv, drop_path=drop_path, correct_mod=correct_mod)
        if skip_last:
            self.edge_dist = _AssertBlock
        else:
            self.edge_dist = MHEDist(dim, num_heads, head_dim, bind_qkv=bind_qkv, drop_path=drop_path, correct_mod=correct_mod)
        
        # FFN, which performs self-updates
        mlp_layer = Split_FFN_GEGLU if split_mode else FFN_GEGLU
        self.edge_ffn = mlp_layer(dim, adj_dim=adj_dim, drop_path=drop_path, islast=skip_last or drop_adjff)
        if (not joint_mlp) and (not skip_last):
            self.node_ffn = mlp_layer(dim, adj_dim=adj_dim, drop_path=drop_path, islast=skip_last or drop_adjff)
        elif skip_last:
            self.node_ffn = _AssertBlock
        else: # otherwise join the MLP, we can keep it in skip_last because it won't be called
            self.node_ffn = self.edge_ffn
            
        # the norm layers
        
        # for self-attention among the nodes
        self.sa_norm = RMSNorm(dim, elementwise_affine=True, eps=1e-6) if use_nodesa else _AssertBlock

        # aggregation 
        self.ca_norm = RMSNorm(dim, elementwise_affine=True, eps=1e-6)
        self.agg_norm = RMSNorm(dim, elementwise_affine=True, eps=1e-6)

        # distribuion
        # self.ca_norm - nodes weren't changed, so we can use the same value
        self.dist_norm = RMSNorm(dim, elementwise_affine=True, eps=1e-6) if not skip_last else _AssertBlock

        # inputs to the FFNs
        self.edge_norm_f = RMSNorm(dim, elementwise_affine=True, eps=1e-6)
        self.edge_norm_a = RMSNorm(adj_dim, elementwise_affine=True, eps=1e-6) if split_mode else _AssertBlock
        
        self.node_norm_f = RMSNorm(dim, elementwise_affine=True, eps=1e-6) if not skip_last else _AssertBlock
        self.node_norm_a = RMSNorm(adj_dim, elementwise_affine=True, eps=1e-6) if split_mode and (not skip_last) else _AssertBlock

        if not shared_adjproj:
            # Note: these will be called by the outer block, but are here for duplication
            self.edge_aproj = AdjProjector(dim, adj_dim, split_mode=split_mode)
            self.node_aproj = AdjProjector(dim, adj_dim, split_mode=split_mode)           

    def enable_save(self, enable):
        self._SAVESTATE = enable

        # always save edge agg
        self.edge_agg.enable_save(enable)
        
        if self.use_nodesa:
            self.node_sa.enable_save(enable)
        # otherwise, no node_sa block
        
        if not self.skip_last:
            self.edge_dist.enable_save(enable)
        # other no edge_dist block
        
        try:
            del self._adj
            del self._adjthr
        except:
            pass

    
    def compute_adj_weights(self, nodes, edges, pop_mask, adj_scale=4.0):
        #print(nodes.shape, edges.shape)
        # nodes is of shape B, nodes + hnodes, C
        # edges is of shape B, edges + hedges, C
        # compute the adjacency weights using mode 1 

        # create the temporary adjacency matrix
        #                 edges  hidden_edges
        #        nodes |                       |
        # hidden_nodes |                       |

        # normalize the inputs to keep the adj values reasonable
        nodes_n = F.normalize(nodes, p=2, dim=-1, eps=1e-6)
        edges_n = F.normalize(edges, p=2, dim=-1, eps=1e-6)

        # compute the dot product
        adj_n = torch.einsum('bij,bkj -> bik', nodes_n, edges_n)
        

        # apply the sigmoid to be used for the penalty
        # this uses the unormalized dot product
        # note: 0.0 means not a member, 1.0 means member
        #adj_sig = F.sigmoid(adj_n * adj_scale)
        # optimize this a bit too
        adj_sig = adj_n.mul_(adj_scale).sigmoid()

        if (self.pop_level is not None) and self.training:
            # Compute the sum of sigmoid activations in each column
            # B,N,E - sum across nodes to produce B,E

            #  population count, number of nodes > 0.5
            # multiply by 2 scale between 0 and 1 rather than 0 and 0.5
            column_sums = 2.0 * F.relu(adj_sig - 0.5).sum(dim=-2)
    
            # positive population penalty, if there are too many nodes
            pop_loss = ( F.relu(column_sums - self.pop_level[0]) ** 1.0).sum(dim=-1)
            # negative population penalty, if there are too few nodes
            pop_loss += ( F.relu(self.pop_level[1] - column_sums) ** 1.0).sum(dim=-1)

            # compute a similarity loss - ended up not using this
            #sim_loss = self.pop_level[2]*cosine_similarity_loss(adj_sig.transpose(-1,-2), pop_mask, first_mean=False)
        
        else:
            pop_loss = 0
        
        # threshold it for binary membership, but we need to invert the mask
        # sig > 0.5 = member, we are shifting to a positive mask, meaning 0=no-member, 1=member
        # we can do this simply by applying a .gt(0.5) operation
        adjthr = adj_sig.gt(0.5).to(nodes.dtype)
        adj_sig = adj_sig.to(nodes.dtype)
        
        return adj_n, adj_sig, adjthr, pop_loss

        
    def forward(self, nodef, nodea, edgef, edgea, ae_mask, ae_mask_p, pop_mask):

        ################################################
        ## STEP 1: Build the adjacency matrix 
        ################################################

        # already using a learned projection in the parent module
        #if not self.split_mode:
        #    # share adjacency and feature vectors
        #    nodea = nodef
        #    edgea = edgef   
        
        # first compute the adjacency matrix
        adj, adjsig, adjthr, pop_loss = self.compute_adj_weights(nodea, edgea, pop_mask)

        # update the adjthr with the ae_mask
        # adjthr here is positive, meaning 0=no-member, 1=member
        # instead of adding, we can simply multiply by the inverse of ae_mask, which becomes 0 where we don't want connections
        # for refrence, adjthr is only every used in the node self attention
        adjthr = adjthr*( 1 - ae_mask)

        # make adjsig contiguous
        adjsig = adjsig.contiguous()

        # adj_sig = soft weight [0,1]. adj_group = hard group [0,1]
        if self._SAVESTATE:
            self._adj = adj
            self._adjsig = adjsig
            self._adjthr = adjthr

        ################################################
        ## STEP 2: Apply self-attention
        ################################################
        
        # self attention - only updates node features: nodef -> nodef
        if self.use_nodesa:
            nodef = nodef + self.node_sa(self.sa_norm(nodef), adjthr)

        ################################################
        ## STEP 3: Perform hyper-edge aggregation
        ################################################      

        # aggregate to edges:  nodef -> edgef
        hnf = self.ca_norm(nodef)
        edgef = edgef + self.edge_agg(hnf, self.agg_norm(edgef), adjsig, ae_mask_p)

        ################################################
        ## STEP 4: Perform hyper-edge point-wise FFN
        ################################################    
        
        # recombine the edges if needed
        # apply the norm to features
        hef = self.edge_norm_f(edgef)
        if self.split_mode:
            # also apply it to the adj
            hea = self.edge_norm_a(edgea)
        else:
            hea = None

        # reshape edge, apply the feed-forward - (edgef, edge_adj) -> (edgef, edge_adj)
        h_ffn = self.edge_ffn(hef, hea)
        if self.split_mode:
            edgef = edgef + h_ffn[0]

            if not (self.skip_last or self.drop_adjff):
                edgea = edgea + h_ffn[1]
            else:
                edgea = None
        else:
            edgef = edgef + h_ffn
            edgea = None
        
        ################################################
        ## STEP 5: Perform hyper-edge distribution
        ################################################    

        # distribute the edges: edgef -> nodef
        #hnf = self.ca_norm(nodef) # don't have to apply again since hnf didn't change
        if not self.skip_last:
            nodef = nodef + self.edge_dist(hnf, self.dist_norm(edgef), adjsig, ae_mask_p)
        # otherwise, we don't want to update nodef since it's not used

        ################################################
        ## STEP 6: Perform node point-wise FFN
        ################################################   
        
        # apply the norm to features
        if not self.skip_last:
            hnf = self.node_norm_f(nodef)
            if self.split_mode:
                # also apply it to the adj
                hna = self.node_norm_a(nodea)
            else:
                hna = None
    
            # reshape nodes, apply the feed-forward - (nodef, node_adj) -> (nodef, node_adj)  
            h_ffn = self.node_ffn(hnf, hna)
            if self.split_mode:
                nodef = nodef + h_ffn[0]
                if not self.drop_adjff:
                    nodea = nodea + h_ffn[1]
                else:
                    nodea = None
            else:
                nodef = nodef + h_ffn
                nodea = None
        # otherwise, skip this stuff because it's not going to be back-prop'd
        

        return nodef, nodea, edgef, edgea, pop_loss

class HypergraphNet(torch.nn.Module):
    def __init__(self, 
                # === I/O params ===
                in_features,             # image input channels (e.g. 3 for RGB)
                image_size,              # input image size (e.g. 224 for ImageNet)
                patch_size,              # input patch size (e.g. 16 for 16x downsample)
                num_classes,             # number of output classes (e.g. 1000 for ImageNet)
                # === Arch params ===
                num_layers,              # number of stacked blocks
                hidden_dim,              # hidden feature dim
                num_heads,               # number of attention heads, with d_k = hidden_dim // num_heads
                hnode_capacity,          # number of virtual nodes
                edge_capacity,           # number of hyperedges
                hedge_capacity,          # number of virtual hyperedges
                adj_dim=None,            # adjacency feature dim, set to hidden_dim if None
                repeat_n=None,           # number of times to repeat the last block (e.g. l=10, r=2 ~ 12 layers)
                cls_head_dim=1024,       # size of the classifier head inner dim
                # === primary state config ===
                split_mode=True,         # split V into (v_a,v_f) vs v_a=v_f   
                shared_adjproj=True,     # when split=False, all layers share a single projection s.t. v_a = v_f*W
                joint_mlp=False,         # same MLP for nodes and edges
                bind_qkv=False,          # bind Q=K=V for SA, and K=V for CA
                use_stem=True,           # uses conv-stem if True, else ise patch embedding if false
                div_hdim=1,              # divide the input hdim for conv-stem by factor 
                use_nodesa=True,         # use SA for nodes, skips SA and only uses CA if False
                # === regularization helpers ===                
                dropout=0.0,             # prediction head class dropout
                drop_path=0.0,           # stochastic path dropout
                drop_decay=True,         # use decayed stochastic droprate if True, else constant for all blocks
                skip_last=True,          # used to skip last blocks with no useful gradient
                weight_init=True,        # Initialize weights with init function
                virtual_init=False,      # initialize the virtual tokens with custom init function
                train_pe=False,          # allow the position embeddings to be trainable vs static sinusoidal 
                use_virtual_pemb=True,   # apply virtual position embeddings for edges and hidden nodes
                use_diversity_loss=True, # use diversity loss, really helps prevent collapse
                output_ln=False,         # use a LN before classifier head if True, else use RMS-Norm
                use_xformers=False,      # use xformers for Vertex SA if true, else use einsum
                correct_mod=False,       # correct the modulation in edge attention
                pop_level=None,          # ignore if None, else (tokens Pmax, tokens PMin, sim weight - not used)
                correct_imgcond=False,   # correct the imagedrop conditioning when using bothA/E
                # == pooling params ===
                pool_type='mean',        # type of pooling to apply: expert, max, mean, image
                expert_topk=2,           # used for expert prediction: number of routes for mean
                expert_ce_weight=0.0,    # used for expert prediction cross-entropy prediction weight
                expert_noise_level=1e-3, # used for expert prediction: noisy logit scaling
                expert_dropout=0.0,      # used for expert prediction: dropout before projection for routes
                expert_label_smoothing=0.1,#used for expert prediction: label smoothing for CE loss
                expert_use_gate=False,   # used for expert prediction: uses a small gated predictor FFN
                expert_use_norm=True,    # used for expert prediction: uses norm for the routers
                # == Position Embeddings ===
                pe_use_interpolation=False,     # allow PE interpolation for larger or smaller resolutions
                pe_interpolate_offset=0.1,      # interpolation offset for PE (DINO used 0.1)
                pe_interpolate_antialias=False, # use antialiasing with PE interpolation (DINO used False)
                pe_latch_pe=False,              # save the PE upon resizing, notably this will overwrite the base PE. be careful!
                 ):
        super(HypergraphNet, self).__init__()

        print(" Env Configs:")
        print(f"\t _ATTN_PRECISION=={_ATTN_PRECISION} {_ATTN_PRECISION == 'fp32'}")
        print(f"\t _USE_APEX=={_USE_APEX}")
        print(f"\t _USE_XFORMERS=={_USE_XFORMERS}")
        print(f"\t _NORMALIZE_QK=={_NORMALIZE_QK}")
        print("---------------------------------------")

        adj_dim = hidden_dim if adj_dim is None else adj_dim
        
        # initialize faster cnns
        torch.backends.cudnn.fastest = True
        torch.backends.cudnn.benchmark = True

        if correct_imgcond:
            img_last = skip_last and ((pool_type == 'image') or (pool_type[:4] == "both"))
        else:
            img_last = skip_last and (pool_type == 'image')
        
        self.use_stem = use_stem
        self.edge_capacity = edge_capacity
        self.hedge_capacity = hedge_capacity
        self.tedge_capacity = edge_capacity + hedge_capacity
        self.hnode_capacity = hnode_capacity
        self.modes = 2 if split_mode else 1
        self.split_mode = split_mode
        self.use_virtual_pemb = use_virtual_pemb
        self.use_diversity_loss = use_diversity_loss
        self.layer_count = num_layers
        self.repeat_n = repeat_n
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.adj_dim = adj_dim
        self.skip_last = skip_last and (not img_last)
        self.pool_type = pool_type
        self.img_pool = (pool_type == 'image')
        self.img_last = img_last
        self.shared_adjproj = shared_adjproj
        # prepare for both pooling
        if pool_type[:4] == "both":
            self.split_pool = True
            output_ln = None # disable layernorm, because it's being applied in the combiner
            self.skipa_last = (repeat_n is None) # if not none, then don't skip
        else:
            self.split_pool = False
            self.skipa_last = False

        self._SAVESTATE = False
        self._SAVEKHOT = False
        self._EVALFLOPS = False
        
        # hyper edges (edge_size), hidden_nodes (node_size), hidden_edges (node_size)
        # if splitmode is False, then hnodea and edgea will have 1 for the first dim, else will have 0 = Null
        self.hnodef = nn.Parameter(torch.randn(1, hnode_capacity, hidden_dim))
        self.hnodea = nn.Parameter(torch.randn(int(split_mode), hnode_capacity, adj_dim), requires_grad=split_mode)
        self.edgef = nn.Parameter(torch.randn(1, edge_capacity + hedge_capacity, hidden_dim))
        self.edgea = nn.Parameter(torch.randn(int(split_mode), edge_capacity + hedge_capacity, adj_dim), requires_grad=split_mode)

        # create the adjacency projections
        if shared_adjproj:
            self.edge_aproj = AdjProjector(hidden_dim, adj_dim, split_mode=split_mode)
            self.node_aproj = AdjProjector(hidden_dim, adj_dim, split_mode=split_mode)
        else:
            self.edge_aproj = None
            self.node_aproj = None

        grid_size = image_size//patch_size
        self.grid_size = grid_size
        self.patch_size = patch_size

        self.pe_use_interpolation = pe_use_interpolation
        self.pe_interpolate_offset = pe_interpolate_offset
        self.pe_interpolate_antialias = pe_interpolate_antialias
        self.pe_latch_pe = pe_latch_pe

        print(f"Constructing hypergraph with")
        print(f" - {grid_size*grid_size} nodes")
        print(f" - {hnode_capacity} hidden nodes")
        print(f" - {edge_capacity} edges")
        print(f" - {hedge_capacity} hidden edges")
        
        # create the embeddings - inputs expected to be in the shape B,H,W,C
        HW = grid_size**2
        self.pos_embed = nn.Parameter(torch.zeros(1, HW, hidden_dim), requires_grad=train_pe)
        
        # initialize
        pos_embed = MaeEmbeddings.get_2d_sincos_pos_embed(hidden_dim, grid_size)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
        pos_embed = rearrange(pos_embed, 'B (H W) C -> B (H W) C', H=grid_size)
        self.pos_embed.data.copy_(pos_embed)

        # compute the maximum virtual dim
        vdim = max(hnode_capacity, edge_capacity + hedge_capacity)

        # create the vitrual positon embeddings
        if use_virtual_pemb:
            # feature virtual embs: note B = 1
            pef = self._create_pos_emb(vdim, hidden_dim)
            pef = rearrange(pef, 'B N C -> B N C')
            self.virtual_pembf = nn.Parameter(torch.zeros_like(pef), requires_grad=train_pe)
            self.virtual_pembf.data.copy_(pef)
            # adjacency virtual embs: note B = 1
            pea = self._create_pos_emb(vdim, adj_dim)
            pea = rearrange(pea, 'B N C -> B N C')
            self.virtual_pemba = nn.Parameter(torch.zeros_like(pea), requires_grad=train_pe)
            self.virtual_pemba.data.copy_(pea)
        else:
            self.virtual_pembf = 0
            self.virtual_pemba = 0

        # build a eye mask for diversity loss
        diag_mask = 1.0 - torch.eye(vdim)
        self.register_buffer('diag_mask', diag_mask)

        # build a population mask for diversity loss
        # pop_mask is tE,tE - we want to mask out the virtual edges
        #                 edges  hidden_edges
        #        edges |   1         0         |
        # hidden_edges |   0         1         |  
        pop_mask = diag_mask.clone()
        pop_mask[edge_capacity:, :edge_capacity] = 0
        pop_mask[:edge_capacity, edge_capacity:] = 0
        self.register_buffer('pop_mask', pop_mask)

        # compute the edge masks 
        mask = torch.zeros(1, HW + self.hnode_capacity, self.tedge_capacity)
        
        # create the aggregate mask 
        #                 edges  hidden_edges
        #        nodes |   0         1         |
        # hidden_nodes |   0         0         |  
        ae_mask = mask.clone()
        ae_mask[:, :HW, self.edge_capacity:] = 1
        self.register_buffer('ae_mask', ae_mask)
        # prescale the ae_mask to avoid excess kernel calls
        self.register_buffer('ae_mask_p', ae_mask*-1e9)

        # the patch embedder
        patcher = ViGStem if self.use_stem else PatchExtractor
        self.patcher = patcher(in_features, hidden_dim, patch_size, split_mode=split_mode, adj_dim=adj_dim, div_hdim=div_hdim)

        print(f"Building transformer with {num_layers} layers, d={hidden_dim}, d_a={adj_dim}, and split_mode={split_mode}")
        
        # create the drop_path rule based on stocahstic decay
        if drop_decay:
            dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)] 
        else:
            dpr = [drop_path]*num_layers

        print("setting stochastic drop_path to:", dpr)

        
        # the hyper-graph transformer
        self.blocks = self.blocks = nn.ModuleList([
            HyperGraphBlock(hidden_dim, 
                            num_heads, 
                            adj_dim=adj_dim,
                            head_dim=-1, 
                            bind_qkv=bind_qkv, 
                            split_mode=split_mode,
                            joint_mlp=joint_mlp,
                            drop_path=dpr[i],
                            pop_level=pop_level,
                            use_xformers=use_xformers,
                            use_nodesa=use_nodesa,
                            shared_adjproj=shared_adjproj,
                            correct_mod=correct_mod,
                            skip_last=self.skip_last and (i == num_layers-1),
                            drop_adjff=(self.img_last or self.skipa_last) and (i == num_layers-1),
                           )
            for i in range(num_layers) ])

        # define the combiner types to build from
        combiners = {
            'expert' : ExpertCombiner, 
            'mean'   : MeanCombiner, 
            'max'    : MaxCombiner,
            'image'  : ImageCombiner,                                                  # ln_norms are img then edge
            'bothA'  : partial(DualCombiner, expert=False, pre_norm=True, drop_path=0.0, ln_norms=[False, False]),
            'bothE'  : partial(DualCombiner, expert=True, pre_norm=True, drop_path=0.0, ln_norms=[False, True]),
            'bothAd'  : partial(DualCombiner, expert=False, pre_norm=True, drop_path=drop_path, ln_norms=[False, False]),
            'bothEd'  : partial(DualCombiner, expert=True, pre_norm=True, drop_path=drop_path, ln_norms=[False, True]),            
        }
        assert pool_type in combiners.keys(), f"Error: unknown {pool_type=}, options are {combiners.keys()}"

            
        print(f"Using Pooling type {pool_type}")
       
        self.combiner = combiners[pool_type](
                                           dim=hidden_dim, 
                                           num_experts=hedge_capacity, 
                                           top_k=expert_topk, 
                                           ce_weight=expert_ce_weight,
                                           label_smoothing=expert_label_smoothing,
                                           noise_level=expert_noise_level,
                                           use_gate=expert_use_gate,
                                           dropout=expert_dropout,
                                           use_norm=expert_use_norm,
                                          )
        
        # the classifier head
        print(f"Creating Classifier with {num_classes} classes and {dropout} dropout and LayerNorm={output_ln}")
        if output_ln is None:
            out_norm = IdentityBlock
        elif output_ln:
            out_norm = LayerNorm
        else:
            out_norm = RMSNorm
            
        self.classifier_head = nn.Sequential(*[
            out_norm(self.combiner.out_dim, elementwise_affine=True, eps=1e-6),
            nn.Linear(self.combiner.out_dim, cls_head_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(cls_head_dim, num_classes, bias=True),
        ])

        if weight_init:
            self._init_weights(virtual_init)

    def _init_weights(self, virtual_init):
        def _basic_init(module):
            # apply to nn.Linear
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            # else apply the module specific weights
            # note that this will prevent traversing within
            elif hasattr(module, 'init_weights'):
                module.init_weights()

        # initialize the linear matricies with trunc norm and zero bias
        # also call module specific module.init_weights
        self.apply(_basic_init)

        # apply zero init fix if flagged
        print("applying zero...")
        # fix the initialization bug - it resets zerod to truncnorm
        for param in self.parameters():
            if hasattr(param, '_zinit') and param._zinit:
                #print("applying zero", param.shape)
                param.detach().zero_()

        # apply virtual init if enabled
        if virtual_init:
            nn.init.normal_(self.hidden_nodes, std=1e-6)
            nn.init.normal_(self.edges, std=1e-6) 



    def _create_pos_emb(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe
    
    def enable_save(self, enable):
        self._SAVESTATE = enable

        for block in self.blocks:
            block.enable_save(enable)

        try:
            del self._patchin
            del self._x
            del self._ei
            del self._khot
            del self._kscores
            del self._pooled_edge
        except:
            pass

    # based on code from DINOv2
    # https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        patch_pos_embed = pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M, f"{N} != {M}^2"
        kwargs = {}
        if self.pe_interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.pe_interpolate_offset) / M
            sy = float(h0 + self.pe_interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.pe_interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed.to(previous_dtype)

    # this function is for use with downstream segmentation tasks
    def forward_img_features(self, x, layer_ids=[-1], flatten=True):
        # extract the relevant shapes
        B,_,HI,WI = x.shape
        
        # make sure save setate is set - this is a hack
        self._SAVESTATE = True
        
        # then call forward like normal
        self.forward(x)

        # now extract the layers
        num_layers = len(self._x)
        extracted = [self._x[i][0][:,:-self.hnode_capacity] if (-num_layers <= i < num_layers) else None for i in layer_ids]

        # stack the elements and reshape into B,C,H,W
        extracted = torch.stack(extracted, dim=0)

        HP = HI//self.patch_size
        WP = WI//self.patch_size

        if flatten:
            return rearrange(extracted, 'L B (HP WP) C -> B (L C) HP WP', HP=HP, WP=HP)
        else:
            return rearrange(extracted, 'L B (HP WP) C -> B L C HP WP', HP=HP, WP=WP)

    
    def forward(self, x, inf_topk=None):
        # extract the relevant shapes
        B,_,HI,WI = x.shape

        # apply the patcher to embed and apply position embeddings to the nodes
        # remove the hyperedge dim, because we aren't using it
        x = self.patcher(x).squeeze(0)

        if self._SAVESTATE:
            self._patchin = x
        
        if self.split_mode:
            # split the adjacency features split mode
            # x is in the shape B,N,C
            ah = x[:,:,self.hidden_dim:]

        # extract HW after the patcher
        HW = x.shape[-2]
        
        # always extract the slice and add pose embeddings
        #x = x[:,:,:self.hidden_dim] + self.pos_embed.to(x.dtype)
        # the above is replaced with the following to support pe interpolation
        x = x[:,:,:self.hidden_dim]
        if self.pe_use_interpolation:
            with torch.no_grad():  # Ensure no gradient tracking
                pe = self.interpolate_pos_encoding(x, WI, HI)
    
                if self.ae_mask.shape[1] != HW + self.hnode_capacity:
                    # we need to rebuild the mask
                    ae_mask = torch.zeros(1, HW + self.hnode_capacity, self.tedge_capacity, device=x.device)
                    ae_mask[:, :HW, self.edge_capacity:] = 1
                    self.register_buffer('ae_mask', ae_mask)
                    # prescale the ae_mask to avoid excess kernel calls
                    self.register_buffer('ae_mask_p', ae_mask*-1e9)
    
                    # update the PE replacing the old ones if true
                    # this is mainly useful for training to avoid constant reinterpolation
                    # however, this will also overwrite the previous PE, which must be
                    # manually deleted from the checkpoint file!!! 
                    if self.pe_latch_pe:
                        if hasattr(self, 'pos_embed'):
                            delattr(self, 'pos_embed')  # Remove existing buffer
                        self.register_buffer('pos_embed', pe)
                
        else:
            pe = self.pos_embed
           
        # finally add the pos embs
        x = x + pe.to(x.dtype)
            
        # apply the vitrual position embeddings if enabled
        if self.use_virtual_pemb:
            xi = self.hnodef + self.virtual_pembf[:,:self.hnode_capacity]
            ei = self.edgef + self.virtual_pembf[:,:self.tedge_capacity]
            if self.split_mode:
                xa = self.hnodea + self.virtual_pemba[:,:self.hnode_capacity]
                ea = self.edgea + self.virtual_pemba[:,:self.tedge_capacity]
        else:
            xi = self.hnodef
            ei = self.edgef
            xa = self.hnodea
            ea = self.edgea
        
        # expand the intial embeddings to the batch dim
        xi = xi.expand(B, -1, -1).to(x.dtype)
        ei = ei.expand(B, -1, -1).to(x.dtype)

        # expand the adj to the batch dim if split
        if self.split_mode:
            xa = xa.expand(B, -1, -1).to(x.dtype)
            ea = ea.expand(B, -1, -1).to(x.dtype)
        else:
            xa = None
            ea = None
      
        # concat the real and virtual vertex features in the sequence dim
        xf = torch.cat([x,xi], dim = 1).contiguous() # concat the real and virtual nodes in the node dim
        # if in split mode, then cat adj too
        if self.split_mode:
            xa = torch.cat([ah,xa], dim = 1).contiguous() # concat the real and virtual nodes in the node dim

        # make sure ei and ea are contiguous
        ef = ei.contiguous()
        if self.split_mode:
            ea = ea.contiguous() 

        # intialize the adj features with a projection or pass them forward
        if self.shared_adjproj:
            xa = self.node_aproj(xfeat=xf, xadj=xa)
            ea = self.edge_aproj(xfeat=ef, xadj=ea)
        else:
            xa = self.blocks[0].node_aproj(xfeat=xf, xadj=xa)
            ea = self.blocks[0].edge_aproj(xfeat=ef, xadj=ea)            
            
        # save the input features if saving is enabled
        if self._SAVESTATE:
            self._x = [(xf, xa)]
            self._ei = [(ef, ea)]

        # compute the total layer count (which includes repeats)
        total_layers = self.layer_count
        if self.repeat_n is not None:
            total_layers += self.repeat_n

        # set an diversity_loss accumulation
        if self.training and self.use_diversity_loss:
            # this one can always take ei, since it's been created
            sim_loss = 0.5*cosine_similarity_loss(ef, self.diag_mask[:self.tedge_capacity,:self.tedge_capacity])
            sim_loss += 0.5*cosine_similarity_loss(ea, self.diag_mask[:self.tedge_capacity,:self.tedge_capacity])
            diversity_loss = sim_loss
            
            # penalize the nodes - slice out hnode^2
            # similarly, this one can always take x since it's been created
            # remember, xf,xa is of shape B,N,C

            sim_loss = 0.5*cosine_similarity_loss(xf[:,HW:], self.diag_mask[:self.hnode_capacity, :self.hnode_capacity])
            sim_loss += 0.5*cosine_similarity_loss(xa[:,HW:], self.diag_mask[:self.hnode_capacity, :self.hnode_capacity])
            diversity_loss = diversity_loss + sim_loss
            diversity_loss = diversity_loss*total_layers # up-weight the initial diveristy loss
        else:
            diversity_loss = 0.0

        # init the population loss to 0
        population_loss = 0.0

        # convert to dtype
        pop_mask = self.pop_mask.to(x.dtype)
        
        # iterate over all blocks     
        for i, block in enumerate(self.blocks):
            # we need to make sure to project the adj tensors each block with the exception of 0, 
            # because we already did it
            # check for skipping the last block
            skip_last = self.skip_last and (i == self.layer_count-1)
            img_last = self.img_last and (i == self.layer_count-1)
            skipa_last = self.skipa_last and (i == self.layer_count-1)
            
            # aply the block
            xf, xa, ef, ea, pl = block(nodef=xf, nodea=xa, edgef=ef, edgea=ea,
                             ae_mask=self.ae_mask, ae_mask_p=self.ae_mask_p, pop_mask=pop_mask)

            # adj projectors get applied at the end of a layer
            if self.shared_adjproj:
                xa = self.node_aproj(xfeat=xf, xadj=xa)
                ea = self.edge_aproj(xfeat=ef, xadj=ea)  
            elif i < (self.layer_count-1):
                xa = self.blocks[i+1].node_aproj(xfeat=xf, xadj=xa)
                ea = self.blocks[i+1].edge_aproj(xfeat=ef, xadj=ea)  
            elif self.repeat_n is not None:
                xa = self.blocks[-1].node_aproj(xfeat=xf, xadj=xa)
                ea = self.blocks[-1].edge_aproj(xfeat=ef, xadj=ea)                 
            else:
                xa = None
                ea = None
            
            population_loss = population_loss + pl # uniform population loss
            
            if self._SAVESTATE:
                self._x.append((xf, xa))
                self._ei.append((ef, ea))
                
            if self.training and self.use_diversity_loss:
                sim_loss = 0.5*cosine_similarity_loss(ef, self.diag_mask[:self.tedge_capacity,:self.tedge_capacity])
                if not (skip_last or img_last or skipa_last): # don't apply diversity to the last adj feature set, since it's not used again
                    sim_loss += 0.5*cosine_similarity_loss(ea, self.diag_mask[:self.tedge_capacity,:self.tedge_capacity])
                    
                diversity_loss = diversity_loss + sim_loss
                
                # remember, xf,xa is of shape B,N,C
                if not (skip_last or img_last): # don't compute diversity on nonsencial values
                    sim_loss = 0.5*cosine_similarity_loss(xf[:,HW:], self.diag_mask[:self.hnode_capacity, :self.hnode_capacity])
                    if not skipa_last:
                        sim_loss += 0.5*cosine_similarity_loss(xa[:,HW:], self.diag_mask[:self.hnode_capacity, :self.hnode_capacity])
                    diversity_loss = diversity_loss + sim_loss

        if self.repeat_n is not None:
            # repeat the last block N times
            for i in range(self.repeat_n):
                skip_last = (not self.img_pool) and (i == self.repeat_n-1)
                img_last = (self.img_pool or self.skipa_last) and (i == self.repeat_n-1)
                # aply the block
                xf, xa, ef, ea, pl = self.blocks[-1](nodef=xf, nodea=xa, edgef=ef, edgea=ea,
                             ae_mask=self.ae_mask, ae_mask_p=self.ae_mask_p, pop_mask=pop_mask)
                
                # adj projectors get applied at the end of a layer
                if self.shared_adjproj:
                    xa = self.node_aproj(xfeat=xf, xadj=xa)
                    ea = self.edge_aproj(xfeat=ef, xadj=ea)  
                else:
                    xa = self.blocks[-1].node_aproj(xfeat=xf, xadj=xa)
                    ea = self.blocks[-1].edge_aproj(xfeat=ef, xadj=ea)                      

                population_loss = population_loss + pl # uniform population loss

                if self._SAVESTATE:
                    self._x.append((xf, xa))
                    self._ei.append((ef, ea))

                if self.training and self.use_diversity_loss:
                    # cat to ei and x
                    sim_loss = 0.5*cosine_similarity_loss(ef, self.diag_mask[:self.tedge_capacity,:self.tedge_capacity])
                    if not (skip_last or img_last): # don't apply diversity to the last adj feature set, since it's not used again
                        sim_loss += 0.5*cosine_similarity_loss(ea, self.diag_mask[:self.tedge_capacity,:self.tedge_capacity])
                        
                    diversity_loss = diversity_loss + sim_loss
                    
                    # remember, xf,xa is of shape B,N,C
                    if not (skip_last or img_last): # don't compute diversity on nonsencial values
                        sim_loss = 0.5*cosine_similarity_loss(xf[:,HW:], self.diag_mask[:self.hnode_capacity, :self.hnode_capacity])
                        sim_loss += 0.5*cosine_similarity_loss(xa[:,HW:], self.diag_mask[:self.hnode_capacity, :self.hnode_capacity])
                        diversity_loss = diversity_loss + sim_loss

        # adjust the diversity_loss based on the layer count
        diversity_loss = diversity_loss / (2*total_layers) # init heavy diversity
        population_loss = population_loss / (total_layers) # uniform population
        
        # now pool only the hidden_edges
        expert_loss = 0.0
        khot = None
        kscores = None

        # perform the pooling function. For edges, only take the hidden edges. For nodes, only take the image nodes
        # remember, edges and nodes are of shape B,N,C
        pooled_edge, khot, kscores, expert_loss = self.combiner( xf[:,:-self.hnode_capacity],
                                                                 ef[:,self.edge_capacity:], inf_topk=inf_topk)
        pooled_edge = pooled_edge.squeeze(1)

        if self._SAVESTATE or self._SAVEKHOT:
            self._pooled_edge = pooled_edge
            self._khot = khot
            self._kscores = kscores

        # and apply the classifier head
        prediction = self.classifier_head(pooled_edge)

        if self._EVALFLOPS:
            return prediction
        
        return prediction, diversity_loss, population_loss, expert_loss