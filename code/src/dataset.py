import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset, BatchSampler


class SequenceBucketCollator():
    def __init__(self, choose_length, maxlen, sequence_index, length_index, label_index=None):
        self.choose_length = choose_length
        self.sequence_index = sequence_index
        self.length_index = length_index
        self.maxlen = maxlen
        self.label_index = label_index

    def __call__(self, batch):
        batch = [torch.stack(x) for x in list(zip(*batch))]

        sequences = batch[self.sequence_index]
        lengths = batch[self.length_index]

        length = self.choose_length(lengths)
        mask = torch.arange(start=self.maxlen, end=0, step=-1) < length
        padded_sequences = sequences[:, mask]

        batch[self.sequence_index] = padded_sequences

        if self.label_index is not None:
            return [x for i, x in enumerate(batch) if i != self.label_index], batch[self.label_index]

        return batch


def prepare_data_loader(X, lengths, batch_size, maxlen, n_workers=4, y=None, shuffle=False):
    if y is None:
        dataset = TensorDataset(torch.from_numpy(X),
                                torch.from_numpy(lengths))
        collator = SequenceBucketCollator(lambda lenghts: lenghts.max(),
                                          maxlen,
                                          sequence_index=0,
                                          length_index=1)
    else:
        dataset = TensorDataset(torch.from_numpy(X),
                                torch.from_numpy(lengths),
                                torch.tensor(y, dtype=torch.float32))
        collator = SequenceBucketCollator(lambda lenghts: lenghts.max(),
                                          maxlen,
                                          sequence_index=0,
                                          length_index=1,
                                          label_index=2)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator, num_workers=n_workers)


class LenMatchBatchSampler(BatchSampler):
    def __iter__(self):

        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx][0] == 0)
            count_zeros = int(count_zeros / 64)
            if len(buckets[count_zeros]) == 0:  buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, "produced an inccorect number of batches. expected %i, but yielded %i" % \
                                     (len(self), yielded)
