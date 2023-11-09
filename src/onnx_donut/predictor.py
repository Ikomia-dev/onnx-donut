import onnxruntime
import os
from transformers import XLMRobertaTokenizerFast, MinLengthLogitsProcessor
import numpy as np
from torchvision.transforms.functional import rotate, resize
from PIL import Image, ImageOps

from torchvision import transforms
import re
import json


class OnnxPredictor:
    def __init__(self, export_folder, sess_options=None, providers=None):
        self.encoder = onnxruntime.InferenceSession(os.path.join(export_folder, 'encoder.onnx'), sess_options,
    def __init__(self, model_folder, sess_options=None, providers=None):
        self.encoder = onnxruntime.InferenceSession(os.path.join(model_folder, 'encoder.onnx'), sess_options,
                                                    providers=providers)

        self.decoder = onnxruntime.InferenceSession(os.path.join(model_folder, 'decoder.onnx'), sess_options,
                                                    providers=providers)

        self.decoder_with_past = onnxruntime.InferenceSession(os.path.join(model_folder, 'decoder_with_past.onnx'),
                                                              sess_options, providers=providers)
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_folder)

        with open(os.path.join(model_folder, "config.json"), 'r') as f:
            config = json.load(f)

        self.image_size = config['input_size']
        self.align_long_axis = config['do_align_long_axis']
        self.mean = config['image_mean']
        self.std = config['image_std']
        self.pad = config['do_pad']
        self.max_length = config['max_length']
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def prepare_input(self, img):
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        input_size = self.image_size
        if self.align_long_axis and (
                (input_size[0] > input_size[1] and img.width > img.height)
                or (input_size[0] < input_size[1] and img.width < img.height)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(input_size))
        img.thumbnail((input_size[1], input_size[0]))
        delta_width = input_size[1] - img.width
        delta_height = input_size[0] - img.height
        if self.pad:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return np.array(self.to_tensor(ImageOps.expand(img, padding)))[None, :]

    def generate(self, img, prompt, max_length=None):
        if max_length is None:
            max_length = self.max_length
        scores = ()
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        input_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="np").input_ids.astype(
            dtype='int32')

        # keep track of which sequences are already finished
        unfinished_sequences = np.ones(1, dtype='int32')

        logits_processor = MinLengthLogitsProcessor(min_length=0, eos_token_id=eos_token_id)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = np.array(eos_token_id) if eos_token_id is not None else None

        encoder_input_ids = self.prepare_input(Image.fromarray(img))

        out_encoder = self.encoder.run(None, {'pixel_values': encoder_input_ids})[0]

        past_key_values = None
        stop = False

        while not stop:
            if past_key_values is None:
                out_decoder = self.decoder.run(None, {'input_ids': input_ids, 'encoder_hidden_states': out_encoder})
                logits = out_decoder[0]
                past_key_values = {'past_key_value_input_' + str(k): out_decoder[k + 1] for k in
                                   range(len(out_decoder[1:]))}

            else:
                out_decoder = self.decoder_with_past.run(None, {'input_ids': input_ids[:, -1:],
                                                                                **past_key_values})
                logits = out_decoder[0]
                past_key_values = {'past_key_value_input_' + str(i): pkv for i, pkv in enumerate(out_decoder[1:])}
            next_token_logits = logits[:, -1, :]

            next_tokens_scores = logits_processor(input_ids, next_token_logits)
            # argmax
            next_tokens = np.argmax(next_tokens_scores, axis=-1).astype(dtype='int32')
            scores += (next_tokens_scores,)

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences * (
                        np.tile(next_tokens, len(eos_token_id_tensor)) != np.prod(eos_token_id_tensor, axis=0))
                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    stop = True

            if len(input_ids[0]) >= max_length:
                stop = True

            # update generated ids, model inputs, and length for next step
            input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=1)

        seq = self.tokenizer.batch_decode(input_ids)[0]
        seq = seq.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        return self.token2json(seq, self.tokenizer)

    @staticmethod
    def token2json(tokens, tokenizer, is_inner_value=False):
        """
        Convert a (generated) token sequence into an ordered JSON format
        """
        output = dict()

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            end_token = re.search(fr"</s_{key}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = OnnxPredictor.token2json(content, tokenizer, is_inner_value=True)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if (
                                    leaf in tokenizer.get_added_vocab()
                                    and leaf[0] == "<"
                                    and leaf[-2:] == "/>"
                            ):
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + OnnxPredictor.token2json(tokens[6:], tokenizer, is_inner_value=True)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}

