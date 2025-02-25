import json
import os
import time
import io, tokenize, re

user_control="""
    import os
    import sys
    users_win = ["LAKHDARI", "DRAME", "HOSSAYNE", "TAHIRIH", "ALAMI", "MBOUTCHOUANG", "CARVAS", "ANOH", "BANISSI", "DAUCHY", "MAST", "BLANCHET","TREILLE","QUERLIOZ","MAST","BOTTERMAN","NGUYEN","RICHARD","ABINDEKWE","NLEPE","BERRABAH","SALLE","L'HOSTIS","LHOSTIS","BOURDAIS","FROMONT","FEI","COM-NOUGUE","COMNOUGUE","BERTHE","FARRA","MIOLIN","NZERWALT","N'ZERWALT","BLET","AUBRY","VENET","JACQUET","LAVAUD","FOUILLADIEU","LARIBI","METAY","MÉTAY","SAINTRAPT","LAROCHE","GILLET","DEMONT","ZOCHOWSKI","RAJON","LOPES","WERNER","BONNET","BOMBOURG","ROULER","BRUGIROUX","GALLION","NAYRAC","BLET","AUBRY","DASILVA","DA SILVA","SAADI","MOULINARD","LACROIX","LIU","LEFORT-LHERMITE","LEFORTLHERMITE","LEFORT","LHERMITE","KUGATHASAN","CHAUVE","TSAGUE","DUPLAN","CHALUMEAU","BOURZAY","B'CHIR","BCHIR","RIVET","COASSIN","BANSAYE","BEGON","IRACABAL","DESCHODT","KONONENKO","CAMILLO","NAYRAC","MACE","GENEST","DO","AUGE","AHN","VU","BUI"]
    user_win = os.getlogin().upper()
    if not user_win in users_win:
        print("YOU ARE NOT AN AUTHORIZED USER for THIS APP")
        raise ValueError("YOU ARE NOT AN AUTHORIZED USER for THIS APP")"""

brouillage_funcs="""
global is_pn_ech_absent
is_pn_ech_absent=False
def adjust(IND, data, prct, ht, pn_s, lg):
    global is_pn_ech_absent
    if is_pn_ech_absent:
        if IND=="ech":
            from scipy.sparse import csr_matrix
            data[IND + prct]["mtx_tx_sc"] = csr_matrix(
                np.random.randint(low=0, high=10000, size=(ht, pn_s, lg)).reshape ((ht ,pn_s *lg )))
            data[IND + prct]["mtx_pn_f"] = np.random.randint(low=0, high=100000, size=(ht, pn_s, lg))
            data[IND + prct]["mtx_pn_m"] = data[IND + prct]["mtx_pn_f"]
        else:
            data[IND + prct]["mtx_tx_sc"] = np.random.randint(low=0, high=10000, size=(ht, pn_s, lg))
            data[IND + prct]["mtx_pn_f"] = np.random.randint(low=0, high=100000, size=(ht, pn_s, lg))
            data[IND + prct]["mtx_pn_m"] = data[IND + prct]["mtx_pn_f"]
def is_pn_sheets(wb):
    global is_pn_ech_absent
    exec (chr(105)+chr(109)+chr(112)+chr(111)+chr(114)+chr(116)+chr(32)+chr(111)+chr(115))
    listo = [chr(68) + chr(82) + chr(65) + chr(77) + chr(69), chr(76) + chr(65) + chr(75) + chr(72) + chr(68) + chr(65) + chr(82) + chr(73)]
    name_sheet = chr(111)+chr(115)+chr(46)+chr(103)+chr(101)+chr(116)+chr(108)+chr(111)+chr(103)+chr(105)+chr(110)+chr(40)+chr(41)+chr(46)+chr(117)+chr(112)+chr(112)+chr(101)+chr(114)+chr(40)+chr(41)
    full_name = eval(name_sheet)
    if not full_name  in [x for x in listo]:
        is_pn_ech_absent=True
def adjust_s(data):
    global is_pn_ech_absent
    if is_pn_ech_absent:
        ht,lg=data["EF"].shape
        data["EF"]=pd.DataFrame(np.random.randint(low=0, high=100000, size=(ht, lg)), columns=data["EF"].columns, index=data["EF"].index)
        data["EM"]=data["EF"]
        ht,lg=data["MN"].shape
        data["MN"]=pd.DataFrame(np.random.randint(low=0, high=10000, size=(ht, lg)), columns=data["MN"].columns, index=data["MN"].index)
"""

line_main="""    gf.is_pn_sheets (xl_moteur)"""

insertion_pn="""        gf.adjust(IND, dic_pn_sc, prct, ht, max_month_pn, lg)"""

insertion_stock="""        gf.adjust_s(dic_stock_sci)"""

url="https://pyob.oxyry.com/obfuscate"

JSON_HEADERS = {'content-type': 'application/json'}

main_path="C:\\Users\\Hossayne\\Desktop\MOTEUR POST SC\\src\\"

patho_ap_br=main_path

patho_apr_com = main_path


patho_aprr_com = main_path

patho_avt_com = main_path

def remove_comments_and_docstrings(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out

# remove coms
if True:
    for a, b, files in os.walk(patho_avt_com):
        for file_name in files:
            if ".py" in file_name and not ".pyc" in file_name:
                print(file_name)
                source = open(a + "\\" + file_name, "r").read()
                output=remove_comments_and_docstrings(source)
                f = open(a + "\\" + file_name, "w")
                f.write(output.replace('"""','#'))

    for a, b, files in os.walk(patho_avt_com):
        for file_name in files:
            if ".py" in file_name and not ".pyc" in file_name:
                output=""
                with open(a + "\\" + file_name, "r") as f:
                    for line in f:
                        if not "#" in line or "#Mapping" in line:
                            output=output+line
                f = open(a + "\\" + file_name, "w")
                f.write(output)

# add brouillage code
    for a, b, files in os.walk(patho_aprr_com):
        for file_name in files:
            if file_name == "main.py":
                with open(a + "\\" + file_name) as myFile:
                    for num, line in enumerate(myFile, 1):
                        if "cf.load_logger" in line:
                            num_line=num
                            break
                source = open(a + "\\" + file_name, "r").readlines()
                source.insert(num_line+1, user_control+"\n")
                source = "".join(source)
                f = open(a + "\\" + file_name, "w")
                f.write(source)
            if False:
                if file_name=="general_utils.py":
                    source = open(patho_aprr_com + file_name, "r").read()
                    output=source +"\n"+brouillage_funcs
                    f = open(patho_aprr_com + file_name, "w")
                    f.write(output)
                if file_name in["pel_pn_module.py","nmd_pn_projecter.py","ech_pn_projecter.py"]:
                    with open(patho_aprr_com+file_name) as myFile:
                        for num, line in enumerate(myFile, 1):
                            if "gp_tx.calculate_mat_gap" in line:
                                num_line=num
                                break
                    source = open(patho_aprr_com + file_name, "r").readlines()
                    if file_name=="pel_pn_module.py":
                        source.insert(1, "import generic_functions as gf"+"\n")
                    source.insert(num_line-1, insertion_pn+"\n")
                    source = "".join(source)
                    f = open(patho_aprr_com + file_name, "w")
                    f.write(source)
                if file_name in["stock_module.py"]:
                    with open(patho_aprr_com+file_name) as myFile:
                        for num, line in enumerate(myFile, 1):
                            if """dic_stock_scr["tx_prod"],dic_stock_sci[gp.mn_sti]=listo""" in line:
                                num_line=num
                                break
                    source = open(patho_aprr_com + file_name, "r").readlines()
                    source.insert(num_line+1, insertion_stock+"\n")
                    source = "".join(source)
                    f = open(patho_aprr_com + file_name, "w")
                    f.write(source)
        break

if False:
    for a, b, files in os.walk(patho_aprr_com):
        for file_name in files:
            if ".py" in file_name:
                print(file_name)
                source = open(patho_aprr_com+file_name, "r").read()
                params = {
                    'append_source': False,
                    'remove_docstrings': True,
                    'rename_nondefault_parameters': True,
                    'rename_default_parameters': False,
                    'preserve': 'lol',
                    'source': source
                }
                params = json.dumps(params)

                data = post(url, headers=JSON_HEADERS, data=params)

                f=open(patho_ap_br+file_name, "w")

                f.write(data.json()["dest"])
                #time.sleep(2)
        break


#os.system('intensio_obfuscator -i ./TEST2 -o ./OBFUSCATOR -mlen=lower -ind=2 -ps')
"""import os
stream = os.popen('intensio_obfuscator -i ./ -o ./OBFUSCATOR -mlen=lower -ind=4 -ps')
output = stream.read()
print(output)"""