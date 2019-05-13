from utils import AttnParams

def defaultParams(model_size='medium', bottleneck='avg', attn_heads=8, attn_mech = "KQV",latent_dim=96, NoFE=False, FILM=False, WAE=False):
    params = AttnParams()
    params["kl_pretrain_epochs"] = 2
    params["kl_anneal_epochs"] = 5
    params["bottleneck"] = bottleneck
    params["stddev"] = 1
    params["decoder"] = "TRANSFORMER"
    params["latent_dim"] = 96

    # handle attention bottleneck
    if bottleneck == "avg" or bottleneck == "sum":
        model_name = params["bottleneck"]
        if params["bottleneck"] == "sum": params["AM_softmax"] = False
        params["bottleneck"] = "attention"
        params["AM_heads"] = attn_heads
        if attn_heads != 1:
            model_name += "-mh" + str(attn_heads)

            params["AM_type"] = attn_mech
            model_name += "-" + attn_mech
    else:
        model_name = bottleneck + "_" + model_size

    if NoFE and FILM:
        print("Cannot use custom enc-dec AND FiLM layers!")
        print("Using only custom enc-dec")
        FILM = False

    if NoFE:
        params["decoder"] = "TRANSFORMER_NoFE"
        model_name+="_NoFE"
    if FILM:
        params["decoder"] = "TRANSFORMER_FILM"
        model_name+="_FILM"

    if model_size == "small":
        # AVG1:     47741
        # AVG2:     53216
        # GRU:      51076
        # GRU_ATTN: 50806
        # CONV:     51629
        # AR_SLIM:  58394

        params["d_model"] = 32
        params["d_inner_hid"] = 196
        params["d_k"] = 6
        params["heads"] = 6
        params["layers"] = 2

        if params["bottleneck"] == "ar_slim":
            params["ID_layers"] = 3
            params["ID_d_model"] = 6
            params["ID_width"] = 6
            params["ID_d_inner_hid"] = 30
            params["ID_d_k"] = 4
            params["ID_d_v"] = 4
            params["ID_heads"] = 5
        elif "ar" in params["bottleneck"]:
            params["ID_layers"] = 2
            params["ID_d_model"] = 8
            params["ID_d_inner_hid"] = 64
            params["ID_width"] = 4
            params["ID_d_k"] = 6
            params["ID_d_v"] = 6
            params["ID_heads"] = 4
        elif params["bottleneck"] == "gru":
            params["ID_layers"] = 3
            params["ID_d_model"] = 48
        elif params["bottleneck"] == "gru_attn":
            params["ID_layers"] = 3
            params["ID_d_model"] = 42
        elif params["bottleneck"] == "conv":
            params["ID_layers"] = 2  # num layers
            params["ID_d_k"] = 5  # min_filt_size/num
            params["ID_d_model"] = 64  # dense dim

    elif model_size == "medium":
        # AVG1:     171413
        # AVG2:     174488
        # GRU:      172664
        # GRU_ATTN: 171308
        # CONV:     171800
        # AR_SLIM:  184968
        params["d_model"] = 64
        params["d_inner_hid"] = 256
        params["d_k"] = 8
        params["heads"] = 8
        params["layers"] = 3

        if params["bottleneck"] == "ar_slim":
            params["ID_layers"] = 4
            params["ID_d_model"] = 8
            params["ID_width"] = 6
            params["ID_d_inner_hid"] = 64
            params["ID_d_k"] = 4
            params["ID_d_v"] = 4
            params["ID_heads"] = 4
        elif "ar" in params["bottleneck"]:
            params["ID_layers"] = 2
            params["ID_d_model"] = 32
            params["ID_width"] = 4
            params["ID_d_inner_hid"] = 196
            params["ID_d_k"] = 7
            params["ID_d_v"] = 7
            params["ID_heads"] = 5
        elif params["bottleneck"] == "gru_attn":
            params["ID_layers"] = 4
            params["ID_d_model"] = 78
        elif params["bottleneck"] == "gru":
            params["ID_layers"] = 4
            params["ID_d_model"] = 82
        elif params["bottleneck"] == "conv":
            params["ID_layers"] = 4
            params["ID_d_k"] = 8
            params["ID_d_model"] = 156

    elif model_size == "big" or model_size == "large":
        # big avg:      1,131,745
        # big ar_log:   1,316,449
        # big GRU:      1,166,740
        # big CONV:     800k
        params["d_model"] = 128
        params["d_inner_hid"] = 768
        params["d_k"] = 12
        params["heads"] = 12
        params["layers"] = 4

        if "ar" in params["bottleneck"]:
            params["ID_layers"] = 3
            params["ID_d_model"] = 40
            params["ID_width"] = 4
            params["ID_d_inner_hid"] = 256
            params["ID_d_k"] = 8
            params["ID_d_v"] = 8
            params["ID_heads"] = 6
        elif "gru" in params["bottleneck"]:
            params["ID_layers"] = 5
            params["ID_d_model"] = 196
        elif params["bottleneck"] == "conv":
            params["ID_layers"] = 5
            params["ID_d_k"] = 8
            params["ID_d_model"] = 756

    params["d_v"] = params["d_k"]

    model_name +="_d" + str(latent_dim)
    if WAE:
        params["WAE_kernel"] = "IMQ_normal"
        params["kl_max_weight"] = 10
        params["WAE_s"] = 2
        model_name+="_WAE"
    else:
        params["kl_max_weight"] = 1
        model_name+="_VAE"

    params['model'] = model_name
    return params