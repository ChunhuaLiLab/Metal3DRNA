from ModifyName1 import *
from PixelateMetalion import *
from Bio.PDB import Selection
from Bio.PDB.PDBParser import PDBParser
import numpy as np
from Bio.PDB.vectors import Vector
import os
import pandas as pd
import math

ions = []
atom_sum_vector = [0, 0, 0]
atom_center_dic = {}
atom_center = {}
distance_count_list = []
ion_dic = {' MG', ' NA', '  K'}


def generate_res_list(path):
    # 生成所有核苷酸与离子列表
    residue_total_list = [[], []]
    RNA_path = path
    with open(RNA_path) as RNA:
        p = PDBParser(QUIET=True)  # 构建PDB解释器 #
        s = p.get_structure(RNA, RNA)
        """ only consider the first model in PDB file """
        model = s[0]
        residue_list = list(model.get_residues())
        modify_residue_atom_name1(residue_list)
        for res in residue_list:
            if res.resname in DICT_RES_NAME:
                res.resname = DICT_RES_NAME[res.resname]
            if res.resname in {'  A', '  U', '  G', '  C'} \
                    and not check_if_lacking_atoms(res, 'OP2'):
                residue_total_list[0].append(res)
                residue_total_list[1].append(path)
    # samples = np.random.randint(len(residue_total_list[0]), size=len(ion_vector))
    return residue_total_list


def generate_grid_sample(root):
    index_list = []
    vector_list_x = []
    vector_list_y = []
    vector_list_z = []
    residue_total_list = generate_res_list(root)
    count_list = []
    for n in range(len(residue_total_list[0])):
        # print(residue_total_list[0][n].resname)
        for atom in residue_total_list[0][n]:
            vector_list_x.append(atom.get_vector()[0])
            vector_list_y.append(atom.get_vector()[1])
            vector_list_z.append(atom.get_vector()[2])
    Max_X = np.array(vector_list_x).max()
    Min_X = np.array(vector_list_x).min()
    Max_Y = np.array(vector_list_y).max()
    Min_Y = np.array(vector_list_y).min()
    Max_Z = np.array(vector_list_z).max()
    Min_Z = np.array(vector_list_z).min()
    range_X = int((Max_X - Min_X) / 2.5)
    range_Y = int((Max_Y - Min_Y) / 2.5)
    range_Z = int((Max_Z - Min_Z) / 2.5)
    for i in range(range_X + 1):
        for j in range(range_Y + 1):
            for k in range(range_Z + 1):
                sample_list = []
                index = 0
                mark = 0
                temp = [0,0,0]
                temp[0] = Min_X + i * 2.5
                temp[1] = Min_Y + j * 2.5
                temp[2] = Min_Z + k * 2.5
                count = 0
                for res in residue_total_list[0]:
                    distance1 = res['OP2'].get_vector() - temp
                    distance1 = math.sqrt(math.pow(distance1[0], 2)
                                         + math.pow(distance1[1], 2) + math.pow(distance1[2], 2))
                    if distance1 <= 10:
                        count += 1
                        break
                if count > 0:
                    sample_list.append(temp)
                    sample_list.append(residue_total_list[1][0])

                    count_list.append(sample_list)
    return count_list, index_list

