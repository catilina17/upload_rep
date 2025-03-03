models_ech_st = ["a-repo-tf", "a-repo-tv", "p-repo-tf", "p-repo-tv", "a-creqtv", "a-creq", "a-criv", "a-crctz", "a-crctf", "a-crctv",
               "a-intbq-tf", "a-intbq-tv", "p-intbq-tf", "p-intbq-tv", "p-cat-tf", "p-cat-tv",
               "a-autres-tf", "a-autres-tv", "p-autres-tf", "p-autres-tv", "a-crif", "cap_floor",
               "p-swap-tf", "p-swap-tv", "a-swap-tf", "a-swap-tv",
               "a-change-tf", "p-change-tf", "p-security-tf", "p-security-tv", "a-security-tf", "a-security-tv",
                ]

models_ech_pn = ["all_ech_pn"]

models_cap_floor = ["cap_floor"]

models_nmd_st = ["nmd_st"]

models_nmd_pn = ["nmd_pn"]

models_repos = ["a-repo-tf", "a-repo-tv", "p-repo-tf", "p-repo-tv"]

models_names_list = models_ech_st + models_ech_pn + models_nmd_st + models_nmd_pn

batch_size_by_products =  ({x:10000 for x in models_ech_st} | {x:5000 for x in models_ech_pn}
               | {x:40000 for x in models_nmd_st + models_nmd_pn})