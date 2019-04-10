import os
from data import srdata

class de_NTIRE2019(srdata.SRData):
    def __init__(self, args, name='de_NTIRE2019', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(de_NTIRE2019, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(de_NTIRE2019, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(de_NTIRE2019, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'TrainGT')
        self.dir_lr = os.path.join(self.apath, 'TrainLR')
        if self.input_large: self.dir_lr += 'L'

