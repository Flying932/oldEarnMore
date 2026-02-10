T = 0.1
act_lr = 1e-05
act_net = dict(cls_embed=True, depth=1, embed_dim=64, type='ActorMaskSAC')
action_wrapper_method = 'softmax'
agent = dict(
    T=0.1,
    act_lr=1e-05,
    act_net=dict(cls_embed=True, depth=1, embed_dim=64, type='ActorMaskSAC'),
    action_wrapper_method='softmax',
    batch_size=64,
    beta_loss_weight=0.01,
    beta_lr=1e-05,
    clip_grad_norm=3.0,
    cri_lr=1e-05,
    cri_net=dict(cls_embed=True, depth=1, embed_dim=64, type='CriticMaskSAC'),
    criterion=dict(reduction='none', type='MSELoss'),
    device=None,
    gamma=0.99,
    if_use_beta=True,
    if_use_per=False,
    if_use_rep=True,
    max_step=10000.0,
    num_envs=1,
    optimizer=dict(lr=1e-05, params=None, type='AdamW'),
    rep_loss_weight=1.0,
    rep_lr=1e-05,
    rep_net=dict(
        cls_embed=True,
        decoder_depth=1,
        decoder_embed_dim=64,
        decoder_num_heads=8,
        depth=1,
        embed_dim=64,
        embed_type='TimesEmbed',
        feature_size=(
            10,
            102,
        ),
        in_chans=1,
        input_dim=102,
        mask_ratio_max=0.8,
        mask_ratio_min=0.6,
        mask_ratio_mu=0.7,
        mask_ratio_std=0.1,
        mlp_ratio=4.0,
        no_qkv_bias=False,
        norm_pix_loss=False,
        num_heads=4,
        num_stocks=28,
        patch_size=(
            10,
            102,
        ),
        pred_num_stocks=28,
        sep_pos_embed=True,
        t_patch_size=1,
        temporal_dim=3,
        trunc_init=False,
        type='MaskTimeState'),
    repeat_times=8,
    reward_scale=1,
    scheduler=dict(
        cycle_limit=0,
        decay_rate=1.0,
        decay_t=512000,
        gamma=0.1,
        initialize=True,
        lr_min=0.0,
        multi_steps=[
            614400,
            1024000,
            1433600,
        ],
        noise_pct=0.67,
        noise_range_t=None,
        noise_seed=42,
        noise_std=1.0,
        t_in_epochs=False,
        t_initial=1024000,
        t_mul=1.0,
        type='MultiStepLRScheduler',
        warmup_lr_init=1e-08,
        warmup_prefix=False,
        warmup_t=307200),
    soft_update_tau=0.005,
    state_value_tau=0,
    transition_shape=dict(
        action=dict(shape=(
            1,
            29,
        ), type='float32'),
        done=dict(shape=(1, ), type='float32'),
        ids_restore=dict(shape=(
            1,
            28,
        ), type='int64'),
        mask=dict(shape=(
            1,
            28,
        ), type='int32'),
        next_state=dict(shape=(
            1,
            28,
            10,
            102,
        ), type='float32'),
        reward=dict(shape=(1, ), type='float32'),
        state=dict(shape=(
            1,
            28,
            10,
            102,
        ), type='float32')),
    type='AgentMaskSAC')
aux_stocks_path = 'datasets/dj30/aux_stocks_files'
batch_size = 64
beta_loss_weight = 0.01
beta_lr = 1e-05
buffer_size = 2000
cri_lr = 1e-05
cri_net = dict(cls_embed=True, depth=1, embed_dim=64, type='CriticMaskSAC')
criterion = dict(reduction='none', type='MSELoss')
data_path = 'datasets/dj30/features'
dataset = dict(
    aux_stocks_path='datasets/dj30/aux_stocks_files',
    data_path='datasets/dj30/features',
    features_name=[
        'open',
        'high',
        'low',
        'close',
        'kmid2',
        'kup2',
        'klow',
        'klow2',
        'ksft2',
        'roc_5',
        'roc_10',
        'roc_20',
        'roc_30',
        'roc_60',
        'ma_5',
        'ma_10',
        'ma_20',
        'ma_30',
        'ma_60',
        'std_5',
        'std_10',
        'std_20',
        'std_30',
        'std_60',
        'beta_5',
        'beta_10',
        'beta_20',
        'beta_30',
        'beta_60',
        'max_5',
        'max_10',
        'max_20',
        'max_30',
        'max_60',
        'min_5',
        'min_10',
        'min_20',
        'min_30',
        'min_60',
        'qtlu_5',
        'qtlu_10',
        'qtlu_20',
        'qtlu_30',
        'qtlu_60',
        'qtld_5',
        'qtld_10',
        'qtld_20',
        'qtld_30',
        'qtld_60',
        'rank_5',
        'rank_10',
        'rank_20',
        'rank_30',
        'rank_60',
        'imax_5',
        'imax_10',
        'imax_20',
        'imax_30',
        'imax_60',
        'imin_5',
        'imin_10',
        'imin_20',
        'imin_30',
        'imin_60',
        'imxd_5',
        'imxd_10',
        'imxd_20',
        'imxd_30',
        'imxd_60',
        'cntp_5',
        'cntp_10',
        'cntp_20',
        'cntp_30',
        'cntp_60',
        'cntn_5',
        'cntn_10',
        'cntn_20',
        'cntn_30',
        'cntn_60',
        'cntd_5',
        'cntd_10',
        'cntd_20',
        'cntd_30',
        'cntd_60',
        'sump_5',
        'sump_10',
        'sump_20',
        'sump_30',
        'sump_60',
        'sumn_5',
        'sumn_10',
        'sumn_20',
        'sumn_30',
        'sumn_60',
        'sumd_5',
        'sumd_10',
        'sumd_20',
        'sumd_30',
        'sumd_60',
    ],
    labels_name=[
        'ret1',
        'mov1',
    ],
    root=None,
    stocks_path='datasets/dj30/stocks.txt',
    temporals_name=[
        'weekday',
        'day',
        'month',
    ],
    type='PortfolioManagementDataset')
days = 10
decoder_depth = 1
decoder_embed_dim = 64
depth = 1
embed_dim = 64
environment = dict(
    dataset=None,
    days=10,
    end_date=None,
    if_norm=True,
    if_norm_temporal=False,
    initial_amount=1000.0,
    mode='train',
    scaler=None,
    start_date=None,
    transaction_cost_pct=0.001,
    type='EnvironmentPV')
feature_size = (
    10,
    102,
)
horizon_len = 64
if_norm = True
if_norm_temporal = False
if_use_beta = True
if_use_per = False
if_use_rep = True
lr = 1e-05
n_steps_per_episode = 1024
num_envs = 1
num_episodes = 10
num_features = 102
num_stocks = 28
optimizer = dict(lr=1e-05, params=None, type='AdamW')
patch_size = (
    10,
    102,
)
pred_num_stocks = 28
rep_loss_weight = 1.0
rep_lr = 1e-05
rep_net = dict(
    cls_embed=True,
    decoder_depth=1,
    decoder_embed_dim=64,
    decoder_num_heads=8,
    depth=1,
    embed_dim=64,
    embed_type='TimesEmbed',
    feature_size=(
        10,
        102,
    ),
    in_chans=1,
    input_dim=102,
    mask_ratio_max=0.8,
    mask_ratio_min=0.6,
    mask_ratio_mu=0.7,
    mask_ratio_std=0.1,
    mlp_ratio=4.0,
    no_qkv_bias=False,
    norm_pix_loss=False,
    num_heads=4,
    num_stocks=28,
    patch_size=(
        10,
        102,
    ),
    pred_num_stocks=28,
    sep_pos_embed=True,
    t_patch_size=1,
    temporal_dim=3,
    trunc_init=False,
    type='MaskTimeState')
repeat_times = 8
root = None
save_freq = 20
scheduler = dict(
    cycle_limit=0,
    decay_rate=1.0,
    decay_t=512000,
    gamma=0.1,
    initialize=True,
    lr_min=0.0,
    multi_steps=[
        614400,
        1024000,
        1433600,
    ],
    noise_pct=0.67,
    noise_range_t=None,
    noise_seed=42,
    noise_std=1.0,
    t_in_epochs=False,
    t_initial=1024000,
    t_mul=1.0,
    type='MultiStepLRScheduler',
    warmup_lr_init=1e-08,
    warmup_prefix=False,
    warmup_t=307200)
seed = 42
stocks_path = 'datasets/dj30/stocks.txt'
tag = 'mask_sac_nepx10_daysx10_bsx64_bufsx2000_hlx64_edx64_depx1_dedx64_dedepx1_rtx8_lrx1e-05_sdx42_nvx1_actlrx1e-05_crilrx1e-05_replrx1e-05_betlrx1e-05_repwx1.0_betwx0.01_awmxsoftmax_Tx0.1_dj30_dgx0'
temporal_dim = 3
test_end_date = '2021-01-08'
test_start_date = '2019-07-22'
train_start_date = '2007-09-26'
transition = [
    'state',
    'action',
    'mask',
    'ids_restore',
    'reward',
    'done',
    'next_state',
]
transition_shape = dict(
    action=dict(shape=(
        1,
        29,
    ), type='float32'),
    done=dict(shape=(1, ), type='float32'),
    ids_restore=dict(shape=(
        1,
        28,
    ), type='int64'),
    mask=dict(shape=(
        1,
        28,
    ), type='int32'),
    next_state=dict(shape=(
        1,
        28,
        10,
        102,
    ), type='float32'),
    reward=dict(shape=(1, ), type='float32'),
    state=dict(shape=(
        1,
        28,
        10,
        102,
    ), type='float32'))
val_start_date = '2018-01-26'
workdir = 'workdir'
