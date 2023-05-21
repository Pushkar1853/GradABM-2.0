using Pkg
Pkg.add("Flux")
Pkg.add("Zygote")
Pkg.add("Statistics")
using Statistics
using Flux
using Flux: @functor, glorot_uniform
using Zygote

# Embedding Attention Sequence
struct EmbedAttenSeq
    rnn
    attn_layer
    layer_norm
    out_layer
end

function EmbedAttenSeq(dim_seq_in::Int, dim_metadata::Int, rnn_out::Int, dim_out::Int, n_layers::Int, bidirectional::Bool, attn, dropout::Float64, rnn_type::String)
    rnn_layer = rnn_type == "LSTM" ? LSTM : GRU
    rnn = rnn_layer(dim_seq_in, rnn_out ÷ (bidirectional ? 2 : 1), n_layers, bidirectional=bidirectional, dropout=dropout)
    attn_layer = attn(rnn_out, rnn_out, rnn_out)
    layer_norm = LayerNorm(rnn_out)
    out_layer = Chain(Dense(rnn_out + dim_metadata, dim_out, tanh), Dropout(dropout))
    return EmbedAttenSeq(rnn, attn_layer, layer_norm, out_layer)
end

# Decoder
struct DecodeSeq
    embed_input
    attn_combine
    rnn
    out_layer
end

function DecodeSeq(dim_seq_in::Int, dim_metadata::Int, rnn_out::Int, dim_out::Int, n_layers::Int, bidirectional::Bool, dropout::Float64, rnn_type::String)
    rnn_layer = rnn_type == "LSTM" ? LSTM : GRU
    embed_input = Dense(dim_seq_in, rnn_out)
    attn_combine = Dense(2 * rnn_out, rnn_out)
    rnn = rnn_layer(rnn_out, rnn_out ÷ (bidirectional ? 2 : 1), n_layers, bidirectional=bidirectional, dropout=dropout)
    out_layer = Chain(Dense(rnn_out, dim_out, tanh), Dropout(dropout))
    return DecodeSeq(embed_input, attn_combine, rnn, out_layer)
end

# Caliberation Neural Network
struct PowerfulCalibNN
    emb_model
    decoder
    out_layer
    min_values
    max_values
    sigmoid
end

function PowerfulCalibNN(metas_train_dim::Int, X_train_dim::Int, device, training_weeks::Int, hidden_dim::Int, out_dim::Int, n_layers::Int, scale_output::String, bidirectional::Bool)
    emb_model = EmbedAttenSeq(X_train_dim, metas_train_dim, hidden_dim, hidden_dim, n_layers, bidirectional, TransformerAttn, 0.0, "LSTM")
    decoder = DecodeSeq(1, metas_train_dim, hidden_dim, out_dim, 1, bidirectional, 0.0, "LSTM")
    out_layer = Chain(Dense(hidden_dim, hidden_dim ÷ 2), BatchNorm(hidden_dim ÷ 2), leakyrelu, Dense(hidden_dim ÷ 2, out_dim))
    min_values = MIN_VAL_PARAMS[scale_output]
    max_values = MAX_VAL_PARAMS[scale_output]
    sigmoid = σ
    return PowerfulCalibNN(emb_model, decoder, out_layer, min_values, max_values, sigmoid)
end

# implementation of forward-pass
function (m::EmbedAttenSeq)(x, meta)
    latent_seqs, encoder_hidden = m.rnn(x)
    latent_seqs = m.attn_layer(latent_seqs)
    latent_seqs = m.layer_norm(sum(latent_seqs, dims=1))
    out = m.out_layer(vcat(latent_seqs, meta))
    return out, encoder_hidden
end

function (m::DecodeSeq)(Hi_data, encoder_hidden, context)
    inputs = Hi_data'
    h0 = zeros(size(inputs, 2), size(encoder_hidden, 2))
    inputs = m.embed_input(inputs)
    context = repeat(context, outer=(size(inputs, 1), 1, 1))
    inputs = cat(inputs, context, dims=3)
    inputs = m.attn_combine(inputs)
    latent_seqs, _ = m.rnn(inputs, h0)
    latent_seqs = m.out_layer(latent_seqs')
    return latent_seqs
end

function (m::PowerfulCalibNN)(x, meta)
    x_embeds, encoder_hidden = m.emb_model(x', meta)
    time_seq = reshape(range(1, m.training_weeks + WEEKS_AHEAD + 1, length=x_embeds), 1, :)
    Hi_data = (time_seq .- minimum(time_seq)) ./ (maximum(time_seq) - minimum(time_seq))
    emb = m.decoder(Hi_data, encoder_hidden, x_embeds)
    out = m.out_layer(emb)
    out = m.min_values .+ (m.max_values .- m.min_values) .* m.sigmoid(out)
    return out
end