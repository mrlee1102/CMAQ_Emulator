import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_path = sys.argv[1]
src_path = sys.argv[2]
try:
    postfix = sys.argv[3]
except:
    postfix = ''

import math
import tensorflow as tf
model = tf.keras.models.load_model(model_path)

prefix = 'const float layer'
layer_count = 0
with open(f"{src_path}/params.h", "w") as f_param:
    f_param.write(f"#ifndef PARAMS{postfix.upper()}_H\n")
    f_param.write(f"#define PARAMS{postfix.upper()}_H\n\n")

    f_param.write("#define CMAQ_PAR_REGION_NUM           17\n")
    f_param.write("#define CMAQ_PAR_SECTOR_NUM           7\n")
    f_param.write("#define CMAQ_PAR_CTRLMAX_NUM          CMAQ_PAR_REGION_NUM*CMAQ_PAR_SECTOR_NUM\n")
    f_param.write("#define CMAQ_PAR_SPECIES_NUM          6\n")
    f_param.write("#define CMAQ_PAR_ROW_NUM              82\n")
    f_param.write("#define CMAQ_PAR_COL_NUM              67\n\n")

    f_param.write("#define UNET_PAR_ROW_NUM              96\n")
    f_param.write("#define UNET_PAR_COL_NUM              64\n\n")

    for layer_i, layer in enumerate(model.layers):
        params = layer.weights
        name = '_'.join(layer.name.split('_')[:-1])
        if not name:
            name = layer.name
        if name.startswith('batch'):
            name = 'batch_normalization'
        if 'transpose' in layer.name:
            name = 'conv2d_transpose'
        header_prefix = f"{prefix}_{layer_count}_{name}"
        if len(params) == 0: # custom layer, dropout, activation
            continue

        print(f"Parsing Layer {layer_i}: {name} to {src_path}/layer_{layer_count}_params_{name}.h")
        with open(f"{src_path}/layer_{layer_count}_params_{name}.h", "w") as f_header:
            f_header.write(f"#ifndef PARAMS_{name.upper()}_{layer_count}{postfix.upper()}_H\n")
            f_header.write(f"#define PARAMS_{name.upper()}_{layer_count}{postfix.upper()}_H\n")
            f_header.write(f"#include \"params.h\"\n\n")

            if len(params) == 2 and 'conv2d' in name: # convolution
                W, b = params
                if 'transpose' in layer.name:
                    k0, k1, f1, f0 = W.shape
                else:
                    k0, k1, f0, f1 = W.shape
                m_name = [
                    f"UNET_PAR_{layer_count}_{name.upper()}{postfix.upper()}_KERNEL_Y",
                    f"UNET_PAR_{layer_count}_{name.upper()}{postfix.upper()}_KERNEL_X",
                    f"UNET_PAR_{layer_count}_{name.upper()}{postfix.upper()}_IN_FILTER",
                    f"UNET_PAR_{layer_count}_{name.upper()}{postfix.upper()}_OUT_FILTER"]
                m_lines = [
                    f"#define {m_name[0]}    {k0}",
                    f"#define {m_name[1]}    {k1}",
                    f"#define {m_name[2]}   {f0}",
                    f"#define {m_name[3]}  {f1}"]
                h_names = [
                    f"{header_prefix}{postfix}_weight",
                    f"{header_prefix}{postfix}_bias"]

                for h_i, h_name in enumerate(h_names):
                    param = params[h_i]
                    line_count = 0
                    if h_i == 0: # weight
                        if 'transpose' in layer.name:
                            k0, k1, f1, f0 = param.shape
                        else:
                            k0, k1, f0, f1 = param.shape
                        f_header.write(f"{h_name}[{m_name[0]}*{m_name[1]}*{m_name[2]}*{m_name[3]}] = {{\n")
                        for l in range(f1):
                            for k in range(f0):
                                for j in range(k1):
                                    for i in range(k0):
                                        if line_count%10 == 0:
                                            f_header.write("\t")
                                        if 'transpose' in layer.name:
                                            value = param[i,j,l,k]
                                        else:
                                            value = param[i,j,k,l]
                                        if value < 0:
                                            f_header.write(f"{value:.10e}, ")
                                        else:
                                            f_header.write(f" {value:.10e}, ")
                                        if line_count%10 == 9:
                                            f_header.write("\n")
                                        line_count += 1
                        f_header.write("};\n\n")
                    
                    if h_i == 1: # bias
                        param_length = param.shape[0]
                        f_header.write(f"{h_name}[{m_name[-1]}] = {{\n")
                        for l in range(param_length):
                            if l%10 == 0:
                                f_header.write("\t")
                            value = param[l]
                            if value < 0:
                                f_header.write(f"{value:.10e}, ")
                            else:
                                f_header.write(f" {value:.10e}, ")
                            if l%10 == 9:
                                f_header.write("\n")
                        f_header.write("};\n\n")

            elif len(params) == 2 and 'dense' in name: # dense
                W, b = params
                dim0, dim1 = W.shape
                m_name = [
                    f"UNET_PAR_{layer_count}_{name.upper()}{postfix.upper()}_IN_DIM",
                    f"UNET_PAR_{layer_count}_{name.upper()}{postfix.upper()}_OUT_DIM"]
                m_lines = [
                    f"#define {m_name[0]}      {dim0}",
                    f"#define {m_name[1]}     {dim1}"]
                h_names = [
                    f"{header_prefix}{postfix}_weight",
                    f"{header_prefix}{postfix}_bias"]

                for h_i, h_name in enumerate(h_names):
                    param = params[h_i]
                    line_count = 0
                    if h_i == 0: # weight
                        dim0, dim1 = param.shape
                        f_header.write(f"{h_name}[{m_name[0]}*{m_name[1]}] = {{\n")
                        for j in range(dim1):
                            for i in range(dim0):
                                if line_count%10 == 0:
                                    f_header.write("\t")
                                value = param[i,j]
                                if value < 0:
                                    f_header.write(f"{value:.10e}, ")
                                else:
                                    f_header.write(f" {value:.10e}, ")
                                if line_count%10 == 9:
                                    f_header.write("\n")
                                line_count += 1
                        f_header.write("};\n\n")
                    
                    if h_i == 1: # bias
                        param_length = param.shape[0]
                        f_header.write(f"{h_name}[{m_name[-1]}] = {{\n")
                        for l in range(param_length):
                            if l%10 == 0:
                                f_header.write("\t")
                            value = param[l]
                            if value < 0:
                                f_header.write(f"{value:.10e}, ")
                            else:
                                f_header.write(f" {value:.10e}, ")
                            if l%10 == 9:
                                f_header.write("\n")
                        f_header.write("};\n\n")

            elif len(params) == 4: # batch normalization
                g, b, m, v = params
                m_name = [
                    f"UNET_PAR_{layer_count}_{name.upper()}{postfix.upper()}_GAMMA",
                    f"UNET_PAR_{layer_count}_{name.upper()}{postfix.upper()}_BETA",
                    f"UNET_PAR_{layer_count}_{name.upper()}{postfix.upper()}_MEAN",
                    f"UNET_PAR_{layer_count}_{name.upper()}{postfix.upper()}_VARIANCE"]
                m_lines = [
                    f"#define {m_name[0]}        {g.shape[0]}",
                    f"#define {m_name[1]}         {b.shape[0]}",
                    f"#define {m_name[2]}         {m.shape[0]}",
                    f"#define {m_name[3]}     {v.shape[0]}"]
                h_names = [
                    f"{header_prefix}{postfix}_gamma",
                    f"{header_prefix}{postfix}_beta",
                    f"{header_prefix}{postfix}_mean",
                    f"{header_prefix}{postfix}_variance"]

                for h_i, h_name in enumerate(h_names):
                    param = params[h_i]
                    param_length = param.shape[0]
                    f_header.write(f"{h_name}[{m_name[h_i]}] = {{\n")
                    for l in range(param_length):
                        if l%10 == 0:
                            f_header.write("\t")
                        value = param[l]
                        if value < 0:
                            f_header.write(f"{value:.10e}, ")
                        else:
                            f_header.write(f" {value:.10e}, ")
                        if l%10 == 9:
                            f_header.write("\n")
                    f_header.write("};\n\n")

            layer_count += 1

            for m_line in m_lines:
                f_param.write(m_line + '\n')
            f_param.write('\n')

            f_header.write("\n#endif\n")
    
    f_param.write("\n#endif\n")