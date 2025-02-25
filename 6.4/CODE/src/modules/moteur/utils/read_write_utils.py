import numpy as np
def numpy_write(data, file_full_path, nb_mois_out, encoding="latin-1", mode='w', header=True, delimiter=";", i_tx_cli="",
                format_output_tx_cli="", nb_indic="", idx_desc_col_num=[], decimal_symbol=",",):
    if i_tx_cli == "":
        with open(file_full_path, mode, encoding=encoding) as f:
            if header:
                np.savetxt(f, np.array([data.columns]), fmt='%s', delimiter=delimiter)
            for row in np.array(data.copy()):
                nums = (delimiter.join("%.2f" % v for v in row[-nb_mois_out - 1:])).replace(".", decimal_symbol)
                line = delimiter.join(str(row[i]) if not i in list(idx_desc_col_num.keys()) else \
                                          (idx_desc_col_num[i] % row[i]).replace(".", ",") \
                                      for i in range(0, len(row[:-nb_mois_out - 1])))
                f.write(line + delimiter + nums + '\n')
    else:
        with open(file_full_path, mode, encoding=encoding) as f:
            if header:
                np.savetxt(f, np.array([data.columns]), fmt='%s', delimiter=delimiter)
            j = 1
            for row in np.array(data.copy()):
                if j % nb_indic == i_tx_cli:
                    nums = delimiter.join(format_output_tx_cli % v for v in row[-nb_mois_out - 1:]).replace(".",
                                                                                                                    ",")
                else:
                    nums = delimiter.join("%.2f" % v for v in row[-nb_mois_out - 1:]).replace(".", ",")
                line = delimiter.join(str(row[i]) if not i in list(idx_desc_col_num.keys()) else \
                                          (idx_desc_col_num[i] % row[i]).replace(".", ",") \
                                      for i in range(0, len(row[:-nb_mois_out - 1])))
                f.write(line + delimiter + nums + '\n')
                j = j + 1