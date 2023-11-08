import os
import torch
import json
from donut import DonutModel


def export_onnx(src_folder, dst_folder, opset_version=17):
    try:
        os.makedirs(dst_folder)
    except OSError:
        pass

    # Set seed
    torch.manual_seed(0)

    # Load original model
    original_model = DonutModel.from_pretrained(src_folder).eval()

    encoder = original_model.encoder
    decoder = original_model.decoder

    # Get tokenizer
    tokenizer = original_model.decoder.tokenizer  # XLMRobertaTokenizerFast.from_pretrained(src_folder,from_slow=True)
    config = {"do_align_long_axis": original_model.config.align_long_axis,
              "input_size": original_model.config.input_size, "max_length": original_model.config.max_length,
              "eos_token_id": tokenizer.eos_token_id, "do_pad": True, "image_mean": (0.485, 0.456, 0.406),
              "image_std": (0.229, 0.224, 0.225)}
    with open(os.path.join(dst_folder, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Export encoder
    dummy_input_encoder = torch.rand((1, 3, *original_model.config.input_size))

    del original_model

    with torch.no_grad():
        dummy_hidden_states = encoder(dummy_input_encoder)

    input_names = ['pixel_values']

    with torch.no_grad():
        torch.onnx.export(encoder, dummy_input_encoder, os.path.join(dst_folder, "encoder.onnx"),
                          input_names=input_names,
                          verbose=False, opset_version=opset_version)

    input_ids = torch.ones((1, 1), dtype=torch.int32)

    with torch.no_grad():
        decoder_output = decoder(input_ids=input_ids, encoder_hidden_states=dummy_hidden_states, use_cache=True)

    past_key_values = decoder_output.past_key_values
    pkv_input_dynamic_axes = {
        "past_key_value_input_" + str(i): {0: 'batch_size', 2: 'sequence_length'} if i % 4 <= 1 else {0: 'batch_size'}
        for i
        in range(len(past_key_values) * 4)}
    pkv_output_dynamic_axes = {k.replace('input', 'output'): v for k, v in pkv_input_dynamic_axes.items()}

    # Export decoder
    input_decoder_names = ["input_ids", "encoder_hidden_states"]

    dummy_input_decoder = dict(input_ids=input_ids, encoder_hidden_states=dummy_hidden_states, use_cache=True,
                               return_dict=True)

    output_decoder_names = ['logits'] + ['past_key_value_output_' + str(i) for i in range(len(past_key_values) * 4)]

    decoder_dynamic_axes = {'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                            **pkv_input_dynamic_axes,
                            **pkv_output_dynamic_axes,
                            'logits': {0: 'batch_size', 1: 'present_sequence_length'}}

    with torch.no_grad():
        torch.onnx.export(decoder, dummy_input_decoder, f=os.path.join(dst_folder, "decoder.onnx"),
                          input_names=input_decoder_names,
                          output_names=output_decoder_names,
                          dynamic_axes=decoder_dynamic_axes, verbose=False, opset_version=opset_version)

    # Export decoder with past
    decoder_with_past_dynamic_axes = {'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                                      **pkv_input_dynamic_axes,
                                      **pkv_output_dynamic_axes,
                                      'logits': {0: 'batch_size', 1: 'present_sequence_length'}}

    input_decoder_with_past_names = ['input_ids', 'encoder_hidden_states'] + list(pkv_input_dynamic_axes.keys())
    output_decoder_with_past_names = ['logits'] + ['past_key_value_output_' + str(i)
                                                   for i in range(len(past_key_values) * 4)]
    input_ids = input_ids[:, -1:]

    dummy_input_decoder_with_past = dict(input_ids=input_ids, encoder_hidden_states=dummy_hidden_states,
                                         past_key_values=decoder_output.past_key_values)

    with torch.no_grad():
        torch.onnx.export(decoder, dummy_input_decoder_with_past,
                          f=os.path.join(dst_folder, "decoder_with_past.onnx"),
                          input_names=input_decoder_with_past_names,
                          output_names=output_decoder_with_past_names
                          , dynamic_axes=decoder_with_past_dynamic_axes, verbose=False, opset_version=opset_version
                          )

    # TOKENIZER
    tokenizer.save_pretrained(dst_folder)
