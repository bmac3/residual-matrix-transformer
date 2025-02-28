import haliax as hax


def cross_entropy_loss_and_log_normalizers(pred_y, Vocab, target_ids):
    if isinstance(Vocab, str):
        Vocab = pred_y.resolve_axis(Vocab)
    target_y = hax.nn.one_hot(target_ids, Vocab)
    return hax.nn.cross_entropy_loss_and_log_normalizers(pred_y, Vocab, target_y)


def cross_entropy_loss(pred_y, Vocab, target_ids):
    ce_loss, _ = cross_entropy_loss_and_log_normalizers(pred_y, Vocab, target_ids)
    return ce_loss


def z_loss(pred_y, Vocab, target_ids, alpha=1e-4):
    ce_loss, log_z = cross_entropy_loss_and_log_normalizers(pred_y, Vocab, target_ids)
    return ce_loss + alpha * log_z ** 2
