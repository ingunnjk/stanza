"""
A trainer class to handle training and testing of models.
"""

import sys
import logging
import torch
from torch import nn

import transformers
from stanza.models.common.trainer import Trainer as BaseTrainer
from stanza.models.common import utils, loss
from stanza.models.common.foundation_cache import NoTransformerFoundationCache
from stanza.models.pos.model import Tagger
from stanza.models.pos.vocab import MultiVocab

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

logger = logging.getLogger('stanza')

# our standard peft config
PEFT_CONFIG = LoraConfig(inference_mode=False,
                         r=8,
                         target_modules=["query", "value", "output.dense"],
                         lora_alpha=16,
                         lora_dropout=0.1,
                         bias="none")
def unpack_batch(batch, device):
    """ Unpack a batch from the data loader. """
    inputs = [b.to(device) if b is not None else None for b in batch[:8]]
    orig_idx = batch[8]
    word_orig_idx = batch[9]
    sentlens = batch[10]
    wordlens = batch[11]
    text = batch[12]
    return inputs, orig_idx, word_orig_idx, sentlens, wordlens, text

class Trainer(BaseTrainer):
    """ A trainer for training models. """
    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None, device=None, foundation_cache=None):
        if model_file is not None:
            # load everything from file
            self.load(model_file, pretrain, args=args, foundation_cache=foundation_cache)
        else:
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.model = Tagger(args, vocab, emb_matrix=pretrain.emb if pretrain is not None else None, share_hid=args['share_hid'], foundation_cache=foundation_cache)

            # PEFT the model, if needed
            if self.args["peft"] and self.args["bert_model"]:
                # fine tune the bert
                self.args["bert_finetune"] = True
                # peft the lovely model
                self.model.bert_model = get_peft_model(self.model.bert_model, PEFT_CONFIG)
                # because we will save this seperately ourselves within the trainer as PEFT
                # weight loading is a tad different
                self.model.unsaved_modules += ["bert_model"]
                self.model.train()
                self.model.bert_model.train()

        self.model = self.model.to(device)
        self.optimizers = utils.get_split_optimizer(self.args['optim'], self.model, self.args['lr'], betas=(0.9, self.args['beta2']), eps=1e-6, weight_decay=self.args.get('initial_weight_decay', None), bert_learning_rate=self.args.get('bert_learning_rate', 0.0), is_peft=self.args.get("peft", False))

        self.schedulers = {}

        if self.args["bert_finetune"]:
            warmup_scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizers["bert_optimizer"],
                # todo late starting?
                0, self.args["max_steps"])
            self.schedulers["bert_scheduler"] = warmup_scheduler

    def update(self, batch, eval=False):
        device = next(self.model.parameters()).device
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, text = unpack_batch(batch, device)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            [i.zero_grad() for i in self.optimizers.values()]
        loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, text)
        if loss == 0.0:
            return loss

        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])

        [i.step() for i in self.optimizers.values()]
        scheduler = self.schedulers.get("bert_scheduler")
        if scheduler:
            scheduler.step()
        return loss_val

    def predict(self, batch, unsort=True):
        device = next(self.model.parameters()).device
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, text = unpack_batch(batch, device)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained = inputs

        self.model.eval()
        batch_size = word.size(0)
        _, preds = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, text)
        upos_seqs = [self.vocab['upos'].unmap(sent) for sent in preds[0].tolist()]
        xpos_seqs = [self.vocab['xpos'].unmap(sent) for sent in preds[1].tolist()]
        feats_seqs = [self.vocab['feats'].unmap(sent) for sent in preds[2].tolist()]

        pred_tokens = [[[upos_seqs[i][j], xpos_seqs[i][j], feats_seqs[i][j]] for j in range(sentlens[i])] for i in range(batch_size)]
        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)
        return pred_tokens

    def save(self, filename, skip_modules=True):
        model_state = self.model.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if k.split('.')[0] in self.model.unsaved_modules]
            for k in skipped:
                del model_state[k]
        params = {
                'model': model_state,
                'vocab': self.vocab.state_dict(),
                'config': self.args
                }

        if self.args["peft"]:
            params["bert_lora"] = get_peft_model_state_dict(self.model.bert_model)

        try:
            torch.save(params, filename, _use_new_zipfile_serialization=False)
            logger.info("Model saved to {}".format(filename))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.warning(f"Saving failed... {e} continuing anyway.")

    def load(self, filename, pretrain, args=None, foundation_cache=None):
        """
        Load a model from file, with preloaded pretrain embeddings. Here we allow the pretrain to be None or a dummy input,
        and the actual use of pretrain embeddings will depend on the boolean config "pretrain" in the loaded args.
        """
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.error("Cannot load model from {}".format(filename))
            raise
        self.args = checkpoint['config']
        if args is not None: self.args.update(args)
        lora_weights = checkpoint.get('bert_lora')
        if lora_weights:
            self.args["peft"] = True
        if 'bert_model' not in self.args:
            self.args['bert_model'] = None

        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        # load model
        emb_matrix = None
        if self.args['pretrain'] and pretrain is not None: # we use pretrain only if args['pretrain'] == True and pretrain is not None
            emb_matrix = pretrain.emb
        if any(x.startswith("bert_model.") for x in checkpoint['model'].keys()):
            logger.debug("Model %s has a finetuned transformer.  Not using transformer cache to make sure the finetuned version of the transformer isn't accidentally used elsewhere", filename)
            foundation_cache = NoTransformerFoundationCache(foundation_cache)
        self.model = Tagger(self.args, self.vocab, emb_matrix=emb_matrix, share_hid=self.args['share_hid'], foundation_cache=foundation_cache)
        self.model.load_state_dict(checkpoint['model'], strict=False)

        # load lora weights, which is special
        if lora_weights:
            self.model.bert_model = get_peft_model(self.model.bert_model, PEFT_CONFIG)
            self.model.unsaved_modules += ["bert_model"]
            set_peft_model_state_dict(self.model.bert_model, lora_weights)


