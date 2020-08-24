import re
import os


class Fimp:
    def __init__(self, f_name=None, num_feat=float("inf"), attr_dict=None, header=None):
        self.f_name = f_name
        self.header = []  # list of lines from the start to the --------- line
        self.table = []  # [[dataset index, name, ranks, relevances], ...]
        self.attrs = {}  # {name: [dataset index, ranks, relevances], ...}
        if f_name is None:
            assert attr_dict is not None
            self.attrs = attr_dict
            self.header = header
            for attr in attr_dict:
                row = [attr_dict[attr][0], attr, attr_dict[attr][1], attr_dict[attr][2]]
                self.table.append(row)
                # self.sort_by_relevance()
        else:
            with open(self.f_name) as f:
                for x in f:
                    self.header.append(x.strip())
                    if x.startswith("---------"):
                        break
                for feat_ind, x in enumerate(f):
                    if feat_ind == num_feat:
                        break
                    ind, name, rnks, rels = x.strip().split("\t")
                    ind = int(ind)
                    rnks = eval(rnks)
                    rels = eval(rels)
                    self.attrs[name] = [ind, rnks, rels]
                    self.table.append([ind, name, rnks, rels])

    def sort_by_attr_index(self):
        self.table.sort(key=lambda row: row[0])

    def sort_by_relevance(self, ranking_index=0):
        self.table.sort(key=lambda row: row[2][ranking_index])

    def get_attr_indices(self):
        return [row[0] for row in self.table]

    def get_attr_names(self):
        return [row[1] for row in self.table]

    def get_relevances(self, ranking_index=None):
        return [
            row[-1] if ranking_index is None else row[-1][ranking_index]
            for row in self.table
        ]

    def get_ranks(self, ranking_index=None):
        return [
            row[-2] if ranking_index is None else row[-2][ranking_index]
            for row in self.table
        ]

    def get_attribute_description(self, attr_name):
        return self.attrs[attr_name]

    def get_nb_rankings(self):
        return 0 if not self.table else len(self.table[0][-1])

    def get_ranking_names(self):
        return [r.strip() for r in self.header[-2].split("\t")[-1][1:-1].split(",")]

    def get_header(self):
        return self.header

    def get_ensemble_size(self):
        for line in self.header:
            if line.startswith("Ensemble size:"):
                return int(line[line.find(":") + 1 :])
        return None

    def get_relief_param(self, line_index):
        assert "Relief" in self.header[0]
        l = self.header[line_index]
        return eval(l[l.find(":") + 1 :].strip())

    def get_relief_neighbours(self):
        return self.get_relief_param(1)

    def get_relief_iterations(self):
        return self.get_relief_param(2)

    def drop_all_except(self, exceptions):
        def project(l):
            return [l[i] for i in do_not_touch_indices]

        def my_eval(s):
            return [e.strip() for e in s[1:-1].split(",")]

        ranking_name_to_index = {
            name: i for i, name in enumerate(self.get_ranking_names())
        }
        do_not_touch_indices = sorted([ranking_name_to_index[n] for n in exceptions])
        # new header line
        first, second, third, fourth = self.header[-2].split("\t")
        third = project(my_eval(third))
        fourth = project(my_eval(fourth))
        self.header[-2] = re.sub(
            "'", "", "{}\t{}\t{}\t{}".format(first, second, third, fourth)
        )
        # new table
        for i in range(len(self.table)):
            for j in [-2, -1]:
                self.table[i][j] = project(self.table[i][j])
        # new attrs
        for a in self.attrs:
            for j in [-2, -1]:
                self.attrs[a][j] = project(self.attrs[a][j])

    def write_to_file(self, out_file):
        with open(out_file, "w", newline="") as f:
            print("\n".join(self.header), file=f)
            for row in self.table:
                ind, name, ranks, rels = row
                print("{}\t{}\t{}\t{}".format(ind, name, ranks, rels), file=f)


def keep_only_some(fimp_in, fimp_out):
    f = Fimp(f_name=fimp_in)
    max_iter = f.get_relief_iterations()[-1]
    f.drop_all_except(["overallIter{}Neigh25".format(max_iter)])
    f.sort_by_relevance()
    f.write_to_file(fimp_out)


def keep_only_some_all():
    for subdir in ["F1", "Accuracy"]:
        for f in os.listdir(subdir):
            fimp = subdir + "/" + f
            print(fimp)
            subdir_new = subdir + "_overallRanking/"
            if not os.path.exists(subdir_new):
                os.makedirs(subdir_new)
            keep_only_some(fimp, subdir_new + f)


# keep_only_some_all()
