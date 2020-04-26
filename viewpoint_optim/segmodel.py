import torch
import torch.nn as nn
import torch.nn.functional as F


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class SegmentationModule(nn.Module):
    _IGNORE_INDEX = 255

    class _MeanFusion:
        def __init__(self, x, classes):
            self.buffer = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
            self.counter = 0

        def update(self, sem_logits):
            probs = F.softmax(sem_logits, dim=1)
            self.counter += 1
            self.buffer.add_((probs - self.buffer) / self.counter)

        def output(self):
            probs, cls = self.buffer.max(1)
            return probs, cls

    class _VotingFusion:
        def __init__(self, x, classes):
            self.votes = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
            self.probs = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))

        def update(self, sem_logits):
            probs = F.softmax(sem_logits, dim=1)
            probs, cls = probs.max(1, keepdim=True)

            self.votes.scatter_add_(1, cls, self.votes.new_ones(cls.size()))
            self.probs.scatter_add_(1, cls, probs)

        def output(self):
            cls, idx = self.votes.max(1, keepdim=True)
            probs = self.probs / self.votes.clamp(min=1)
            probs = probs.gather(1, idx)
            return probs.squeeze(1), cls.squeeze(1)

    class _MaxFusion:
        def __init__(self, x, _):
            self.buffer_cls = x.new_zeros(x.size(0), x.size(2), x.size(3), dtype=torch.long)
            self.buffer_prob = x.new_zeros(x.size(0), x.size(2), x.size(3))

        def update(self, sem_logits):
            probs = F.softmax(sem_logits, dim=1)
            max_prob, max_cls = probs.max(1)

            replace_idx = max_prob > self.buffer_prob
            self.buffer_cls[replace_idx] = max_cls[replace_idx]
            self.buffer_prob[replace_idx] = max_prob[replace_idx]

        def output(self):
            return self.buffer_prob, self.buffer_cls

    def __init__(self, model_dict, head_channels, classes, depth_fusion, vote_mode="plain", vote_scales=[0.7, 1.2]):
        super(SegmentationModule, self).__init__()
        self.depth_fusion = depth_fusion
        self.vote_mode = vote_mode
        self.vote_scales = vote_scales
        self.body = model_dict['body']
        if depth_fusion == 'feature-concat':
            self.depth_body = model_dict['depth_body']
        self.head = model_dict['head']
        self.cls = nn.Conv2d(head_channels, classes, 1)

        self.classes = classes
        if not self.train:
            if "mean" in self.vote_mode:
                self.fusion_cls = SegmentationModule._MeanFusion
            elif "voting" in self.vote_mode:
                self.fusion_cls = SegmentationModule._VotingFusion
            elif "max" in self.vote_mode:
                self.fusion_cls = SegmentationModule._MaxFusion

    def _forward(self, x, depth):
        img_shape = x.shape[-2:]
        if self.depth_fusion == 'pixel-concat':
            x = torch.cat([x, depth], dim=1)
        x = self.body(x)
        if self.depth_fusion == 'feature-concat':
            depth = self.depth_body(depth)
            x = torch.cat([x, depth], dim=1)
        x = self.head(x)
        x = self.cls(x)
        x = F.interpolate(x, size=img_shape, mode='bilinear', align_corners=True)
        return x

    def forward(self, x, depth):
        if self.train or self.vote_mode == 'plain':
            return self._forward(x, depth)
        else:
            # Prepare data_dict
            feed_dict = [{"x": x, "depth": depth}]
            feed_scales = [1]
            for scale in self.vote_scales:
                scaled_size = [round(s * scale) for s in x.shape[-2:]]
                feed_dict.append(
                    {
                        "x": F.interpolate(x, size=scaled_size, mode="bilinear"),
                        "depth": F.interpolate(depth, size=scaled_size, mode="bilinear")
                    })
                feed_scales.append(scale)
            if "flip" in self.vote_mode:
                for i in range(len(feed_scales)):
                    feed_dict.append(
                        {
                            "x": flip(feed_dict[i]["x"], -1),
                             "depth": flip(feed_dict[i]["depth"], -1)
                        })
                    feed_scales.append(-feed_scales[i])

            fusion = self.fusion_cls(x, self.classes)
            for i in range(len(feed_scales)):
                sem_logits = self._forward(x, **feed_dict[i])
                if feed_scales[i] < 0:
                    sem_logits = flip(sem_logits, -1)
                if abs(feed_scales[i]) != 1:
                    sem_logits = F.interpolate(sem_logits, size=x.shape[-2:], mode="bilinear")
                fusion.update(sem_logits)

        return fusion.output()
