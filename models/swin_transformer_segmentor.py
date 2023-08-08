import torch
from torch import nn

from models.swin.decode_head import DecodeHead
from models.swin.transformer_decode_head import TransformerDecodeHead
from models.swin.swin_transformer import SwinTransformer


def check_keywords_in_name(name, keywords=()):
    for keyword in keywords:
        if keyword in name:
            return True
    return False


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


class SwinSeg(nn.Module):

    def __init__(self, channel_list, class_num=1, in_chans=4, depths=(2, 2, 6, 2), ape=False):
        super(SwinSeg, self).__init__()
        self.backbone = SwinTransformer(depths=depths, in_chans=in_chans, ape=ape)
        self.decode_head = DecodeHead(channel_list, class_num=class_num)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, rgb, depth):
        print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr: ", rgb.size(), depth.size())
        feature_list = self.backbone(rgb, depth)
        output_map = self.decode_head(feature_list)
        return output_map


if __name__ == "__main__":
    channel_in_list = [768, 384, 192, 96]
    s_win = SwinSeg(channel_in_list, depths=[2, 2, 18, 2])
    if torch.cuda.is_available():
        s_win.cuda()
    s_win.eval()
    state_dict = torch.load('stats/swin_small_patch4_window7_224_22k.pth',
                            map_location='cpu')['model']

    delete_list = list()

    # for key in state_dict.keys():
    #     # print(state_dict[key])
    #     if 'decode_head' in key or 'attn_mask' in key:
    #         delete_list.append(key)
    # print(delete_list)
    #
    # for del_key in delete_list:
    #     del state_dict[del_key]
    #
    # for key in state_dict.keys():
    #     if 'decode_head' in key:
    #         print("check: ", key)

    # print('\n\n\n\n')
    #
    # for key1, key2 in zip(state_dict.keys(), s_win.backbone.state_dict().keys()):
    #     print(key1, '      ', key2)

    new_dict = dict()
    for key in s_win.backbone.state_dict().keys():
        if key in state_dict.keys():
            tensor1 = state_dict[key]
            tensor2 = s_win.backbone.state_dict()[key]
            if tensor1.size() == tensor2.size():
                new_dict.update({key: state_dict[key]})
            else:
                print("tensor size not match: ", key)
                new_dict.update({key: s_win.backbone.state_dict()[key]})
        else:
            print("key not in pth: ", key)
            new_dict.update({key: s_win.backbone.state_dict()[key]})

    torch.save(new_dict, "stats/swin-small-22k.pth")

    input_tensor = torch.zeros(size=(1, 3, 1024, 512))
    depth_tensor = torch.zeros(size=(1, 1, 1024, 512))
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        depth_tensor = depth_tensor.cuda()
    with torch.no_grad():
        out = s_win(input_tensor, depth_tensor)
    print(out.shape)
