from pathlib import Path
import numpy as np
import pandas as pd
import csv
import time
import matplotlib.pyplot as plt

start_time = time.time()

hours = 6

delay = 5.2
alp = 1
a = 1

jjj = 1

piii = np.arange(-5 * np.pi, 5 * np.pi, 0.025)
piii2 = np.ceil(np.sin(piii))
piii3 = np.ceil(np.cos(piii))

piii22 = np.sin(piii)
piii33 = np.cos(piii)
xx = 1

Nnc = hours * 3600
r = 400 * np.ones(Nnc)
ref = r
time_range = range(Nnc)

in_ref = ref
len1 = len(in_ref)

x_d = in_ref  # desired trajectory
# in Matlab we had d(1) = x_d(1)
time_vector = np.zeros(Nnc)
time_vector_min = np.zeros(Nnc)
time_vector_hour = np.zeros(Nnc)
target_ch1 = np.zeros(Nnc)
target_ch2 = np.zeros(Nnc)
target_ch3 = np.zeros(Nnc)
target_ch4 = np.zeros(Nnc)
target_ch5 = np.zeros(Nnc)
target_ch6 = np.zeros(Nnc)
target_ch7 = np.zeros(Nnc)
target_ch8 = np.zeros(Nnc)

dot_x = np.zeros(Nnc)

error_ch1 = np.zeros(Nnc)  # e
error_ch2 = np.zeros(Nnc)
error_ch3 = np.zeros(Nnc)
error_ch4 = np.zeros(Nnc)
error_ch5 = np.zeros(Nnc)
error_ch6 = np.zeros(Nnc)
error_ch7 = np.zeros(Nnc)
error_ch8 = np.zeros(Nnc)

u_sat_ch1 = np.zeros(Nnc)
u_sat_ch2 = np.zeros(Nnc)
u_sat_ch3 = np.zeros(Nnc)
u_sat_ch4 = np.zeros(Nnc)
u_sat_ch5 = np.zeros(Nnc)
u_sat_ch6 = np.zeros(Nnc)
u_sat_ch7 = np.zeros(Nnc)
u_sat_ch8 = np.zeros(Nnc)

init_input_ch1 = np.zeros(Nnc)  # u
init_input_ch2 = np.zeros(Nnc)
init_input_ch3 = np.zeros(Nnc)
init_input_ch4 = np.zeros(Nnc)
init_input_ch5 = np.zeros(Nnc)
init_input_ch6 = np.zeros(Nnc)
init_input_ch7 = np.zeros(Nnc)
init_input_ch8 = np.zeros(Nnc)

init_artificail_ch1 = np.zeros(Nnc)  # nu
init_artificail_ch2 = np.zeros(Nnc)
init_artificail_ch3 = np.zeros(Nnc)
init_artificail_ch4 = np.zeros(Nnc)
init_artificail_ch5 = np.zeros(Nnc)
init_artificail_ch6 = np.zeros(Nnc)
init_artificail_ch7 = np.zeros(Nnc)
init_artificail_ch8 = np.zeros(Nnc)

s_ch1 = np.zeros(Nnc)
s_ch2 = np.zeros(Nnc)
s_ch3 = np.zeros(Nnc)
s_ch4 = np.zeros(Nnc)
s_ch5 = np.zeros(Nnc)
s_ch6 = np.zeros(Nnc)
s_ch7 = np.zeros(Nnc)
s_ch8 = np.zeros(Nnc)

u_app_ch1 = np.zeros(Nnc)
u_app_ch2 = np.zeros(Nnc)
u_app_ch3 = np.zeros(Nnc)
u_app_ch4 = np.zeros(Nnc)
u_app_ch5 = np.zeros(Nnc)
u_app_ch6 = np.zeros(Nnc)
u_app_ch7 = np.zeros(Nnc)
u_app_ch8 = np.zeros(Nnc)

i_1_value = np.zeros(Nnc)
i_2_value = np.zeros(Nnc)
i_3_value = np.zeros(Nnc)
i_4_value = np.zeros(Nnc)
i_5_value = np.zeros(Nnc)
i_6_value = np.zeros(Nnc)
i_7_value = np.zeros(Nnc)
i_8_value = np.zeros(Nnc)

x_ch1 = np.zeros(Nnc)
x_ch2 = np.zeros(Nnc)
x_ch3 = np.zeros(Nnc)
x_ch4 = np.zeros(Nnc)
x_ch5 = np.zeros(Nnc)
x_ch6 = np.zeros(Nnc)
x_ch7 = np.zeros(Nnc)
x_ch8 = np.zeros(Nnc)

if error_ch1[0] > 0:
    K_pos_ch1 = 0.4  # tunable positive parameter 1
    ro_gain_ch1 = .08  # tunable positive parameter: .2
else:
    K_pos_ch1 = 0.4  # tunable positive parameter 1
    ro_gain_ch1 = .008  # tunable positive parameter: .05

if error_ch2[0] > 0:
    K_pos_ch2 = 0.4  # tunable positive parameter 1
    ro_gain_ch2 = .08  # tunable positive parameter: .2
else:
    K_pos_ch2 = 0.4  # tunable positive parameter 1
    ro_gain_ch2 = .008  # tunable positive parameter: .05

if error_ch3[0] > 0:
    K_pos_ch3 = 0.4  # tunable positive parameter 1
    ro_gain_ch3 = .08  # tunable positive parameter: .2
else:
    K_pos_ch3 = 0.4  # tunable positive parameter 1
    ro_gain_ch3 = .008  # tunable positive parameter: .05

if error_ch4[0] > 0:
    K_pos_ch4 = 0.4  # tunable positive parameter 1
    ro_gain_ch4 = .08  # tunable positive parameter: .2
else:
    K_pos_ch4 = 0.4  # tunable positive parameter 1
    ro_gain_ch4 = .008  # tunable positive parameter: .05

if error_ch5[0] > 0:
    K_pos_ch5 = 0.4  # tunable positive parameter 1
    ro_gain_ch5 = .08  # tunable positive parameter: .2
else:
    K_pos_ch5 = 0.4  # tunable positive parameter 1
    ro_gain_ch5 = .008  # tunable positive parameter: .05

if error_ch6[0] > 0:
    K_pos_ch6 = 0.4  # tunable positive parameter 1
    ro_gain_ch6 = .08  # tunable positive parameter: .2
else:
    K_pos_ch6 = 0.4  # tunable positive parameter 1
    ro_gain_ch6 = .008  # tunable positive parameter: .05

if error_ch7[0] > 0:
    K_pos_ch7 = 0.4  # tunable positive parameter 1
    ro_gain_ch7 = .08  # tunable positive parameter: .2
else:
    K_pos_ch7 = 0.4  # tunable positive parameter 1
    ro_gain_ch7 = .008  # tunable positive parameter: .05

if error_ch8[0] > 0:
    K_pos_ch8 = 0.4  # tunable positive parameter 1
    ro_gain_ch8 = .08  # tunable positive parameter: .2
else:
    K_pos_ch8 = 0.4  # tunable positive parameter 1
    ro_gain_ch8 = .008  # tunable positive parameter: .05

A_max = 3.3  # maximum Voltage to be applied
min_val = 0  # minimum Voltage to be applied

T_samp = 1  # sampling time for the denominator

# -1 if the response is proportional and +1 if it is inverse
sign_flag = -1

aaa_ch1 = init_input_ch1[0]
aaa_ch2 = init_input_ch2[0]
aaa_ch3 = init_input_ch3[0]
aaa_ch4 = init_input_ch4[0]
aaa_ch5 = init_input_ch5[0]
aaa_ch6 = init_input_ch6[0]
aaa_ch7 = init_input_ch7[0]
aaa_ch8 = init_input_ch8[0]

time_table = 5
v1_table = 1
v2_table = 1
v3_table = 1
v4_table = 1
v5_table = 1
v6_table = 1
v7_table = 1
v8_table = 1

# field names
csv_table = {'t(s)': [time_table], 'V1(v)': [v1_table], 'V2(v)': [v2_table], 'V3(v)': [v3_table], 'V4(v)': [v4_table],
             'V5(v)': [v5_table], 'V6(v)': [v6_table], 'V7(v)': [v7_table], 'V8(v)': [v8_table]}

# creating dataframe from the above dictionary of lists
dataFrame = pd.DataFrame(csv_table)

# write dataFrame to SalesRecords CSV file
dataFrame.to_csv('~/Desktop/Close_Loop_Actuation/Wound_1.csv', index=False)

u_app_ch1[0] = aaa_ch1
u_app_ch2[0] = aaa_ch2
u_app_ch3[0] = aaa_ch3
u_app_ch4[0] = aaa_ch4
u_app_ch5[0] = aaa_ch5
u_app_ch6[0] = aaa_ch6
u_app_ch7[0] = aaa_ch7
u_app_ch8[0] = aaa_ch8
df = pd.read_csv('~/Desktop/Close_Loop_Actuation/Output/Wound_1.csv')

m_size = len(df.index)
while m_size == 0:
    print("waiting for a file")
    df = pd.read_csv('~/Desktop/Close_Loop_Actuation/Output/Wound_1.csv')

    m_size = len(df.index)
    time.sleep(10)
print("file found")
m_size_old = len(df.index)

print(df.tail(1))

i_1_value[0] = df.iat[-1, 1]
i_2_value[0] = df.iat[-1, 2]
i_3_value[0] = df.iat[-1, 3]
i_4_value[0] = df.iat[-1, 4]
i_5_value[0] = df.iat[-1, 5]
i_6_value[0] = df.iat[-1, 6]
i_7_value[0] = df.iat[-1, 7]
i_8_value[0] = df.iat[-1, 8]

x_ch1[0] = i_1_value[0]
x_ch2[0] = i_2_value[0]
x_ch3[0] = i_3_value[0]
x_ch4[0] = i_4_value[0]
x_ch5[0] = i_5_value[0]
x_ch6[0] = i_6_value[0]
x_ch7[0] = i_7_value[0]
x_ch8[0] = i_8_value[0]

time_vector[0] = 0

for i in range(1, Nnc):
    if time.time() - start_time > hours * 3600:
        print("The code has been run {} hours".format(hours))
        break
    df = pd.read_csv('~/Desktop/Close_Loop_Actuation/Output/Wound_1.csv')
    m_size = len(df.index)
    time.sleep(4.5)
    row_index = i  # normally it is -1 to read the last row for testing on file that doesnt change, set the value to i
    if row_index < 0:
        while m_size == 0:
            print("waiting for new data")
            df = pd.read_csv('~/Desktop/Close_Loop_Actuation/Output/Wound_1.csv')

            m_size = len(df.index)
            time.sleep(0.5)
    m_size_old = m_size

    time_vector[i] = (time.time() - start_time)
    time_vector_min[i] = (time.time() - start_time) / 60
    time_vector_hour[i] = (time.time() - start_time) / 3600

    i_1_value[i] = df.iat[row_index, 1]
    i_2_value[i] = df.iat[row_index, 2]
    i_3_value[i] = df.iat[row_index, 3]
    i_4_value[i] = df.iat[row_index, 4]
    i_5_value[i] = df.iat[row_index, 5]
    i_6_value[i] = df.iat[row_index, 6]
    i_7_value[i] = df.iat[row_index, 7]
    i_8_value[i] = df.iat[row_index, 8]

    x_ch1[i] = i_1_value[i]
    x_ch2[i] = i_2_value[i]
    x_ch3[i] = i_3_value[i]
    x_ch4[i] = i_4_value[i]
    x_ch5[i] = i_5_value[i]
    x_ch6[i] = i_6_value[i]
    x_ch7[i] = i_7_value[i]
    x_ch8[i] = i_8_value[i]

    error_ch1[i] = x_ch1[i] - x_d[i]
    error_ch2[i] = x_ch2[i] - x_d[i]
    error_ch3[i] = x_ch3[i] - x_d[i]
    error_ch4[i] = x_ch4[i] - x_d[i]
    error_ch5[i] = x_ch5[i] - x_d[i]
    error_ch6[i] = x_ch6[i] - x_d[i]
    error_ch7[i] = x_ch7[i] - x_d[i]
    error_ch8[i] = x_ch8[i] - x_d[i]

    s_ch1[i] = K_pos_ch1 * (error_ch1[i]) + (((x_ch1[i] - x_ch1[i - 1]) / T_samp) - ((r[i] - r[i - 1]) / T_samp))
    s_ch2[i] = K_pos_ch2 * (error_ch2[i]) + (((x_ch2[i] - x_ch2[i - 1]) / T_samp) - ((r[i] - r[i - 1]) / T_samp))
    s_ch3[i] = K_pos_ch3 * (error_ch3[i]) + (((x_ch3[i] - x_ch3[i - 1]) / T_samp) - ((r[i] - r[i - 1]) / T_samp))
    s_ch4[i] = K_pos_ch4 * (error_ch4[i]) + (((x_ch4[i] - x_ch4[i - 1]) / T_samp) - ((r[i] - r[i - 1]) / T_samp))
    s_ch5[i] = K_pos_ch5 * (error_ch5[i]) + (((x_ch5[i] - x_ch5[i - 1]) / T_samp) - ((r[i] - r[i - 1]) / T_samp))
    s_ch6[i] = K_pos_ch6 * (error_ch6[i]) + (((x_ch6[i] - x_ch6[i - 1]) / T_samp) - ((r[i] - r[i - 1]) / T_samp))
    s_ch7[i] = K_pos_ch7 * (error_ch7[i]) + (((x_ch7[i] - x_ch7[i - 1]) / T_samp) - ((r[i] - r[i - 1]) / T_samp))
    s_ch8[i] = K_pos_ch8 * (error_ch8[i]) + (((x_ch8[i] - x_ch8[i - 1]) / T_samp) - ((r[i] - r[i - 1]) / T_samp))

    init_artificail_ch1[i] = sign_flag * ro_gain_ch1 * np.sign(s_ch1[i] * A_max * np.cos(init_input_ch1[i - 1]))
    init_artificail_ch2[i] = sign_flag * ro_gain_ch2 * np.sign(s_ch2[i] * A_max * np.cos(init_input_ch2[i - 1]))
    init_artificail_ch3[i] = sign_flag * ro_gain_ch3 * np.sign(s_ch3[i] * A_max * np.cos(init_input_ch3[i - 1]))
    init_artificail_ch4[i] = sign_flag * ro_gain_ch4 * np.sign(s_ch4[i] * A_max * np.cos(init_input_ch4[i - 1]))
    init_artificail_ch5[i] = sign_flag * ro_gain_ch5 * np.sign(s_ch5[i] * A_max * np.cos(init_input_ch5[i - 1]))
    init_artificail_ch6[i] = sign_flag * ro_gain_ch6 * np.sign(s_ch6[i] * A_max * np.cos(init_input_ch6[i - 1]))
    init_artificail_ch7[i] = sign_flag * ro_gain_ch7 * np.sign(s_ch7[i] * A_max * np.cos(init_input_ch7[i - 1]))
    init_artificail_ch8[i] = sign_flag * ro_gain_ch8 * np.sign(s_ch8[i] * A_max * np.cos(init_input_ch8[i - 1]))

    init_input_ch1[i] = init_input_ch1[i - 1] + (init_artificail_ch1[i - 1] + init_artificail_ch1[i]) / 2 * T_samp
    init_input_ch2[i] = init_input_ch2[i - 1] + (init_artificail_ch2[i - 1] + init_artificail_ch2[i]) / 2 * T_samp
    init_input_ch3[i] = init_input_ch3[i - 1] + (init_artificail_ch3[i - 1] + init_artificail_ch3[i]) / 2 * T_samp
    init_input_ch4[i] = init_input_ch4[i - 1] + (init_artificail_ch4[i - 1] + init_artificail_ch4[i]) / 2 * T_samp
    init_input_ch5[i] = init_input_ch5[i - 1] + (init_artificail_ch5[i - 1] + init_artificail_ch5[i]) / 2 * T_samp
    init_input_ch6[i] = init_input_ch6[i - 1] + (init_artificail_ch6[i - 1] + init_artificail_ch6[i]) / 2 * T_samp
    init_input_ch7[i] = init_input_ch7[i - 1] + (init_artificail_ch7[i - 1] + init_artificail_ch7[i]) / 2 * T_samp
    init_input_ch8[i] = init_input_ch8[i - 1] + (init_artificail_ch8[i - 1] + init_artificail_ch8[i]) / 2 * T_samp

    u_sat_ch1[i] = A_max * np.sin(init_input_ch1[i])
    u_sat_ch2[i] = A_max * np.sin(init_input_ch2[i])
    u_sat_ch3[i] = A_max * np.sin(init_input_ch3[i])
    u_sat_ch4[i] = A_max * np.sin(init_input_ch4[i])
    u_sat_ch5[i] = A_max * np.sin(init_input_ch5[i])
    u_sat_ch6[i] = A_max * np.sin(init_input_ch6[i])
    u_sat_ch7[i] = A_max * np.sin(init_input_ch7[i])
    u_sat_ch8[i] = A_max * np.sin(init_input_ch8[i])

    aaa_ch1 = u_sat_ch1[i]
    aaa_ch2 = u_sat_ch2[i]
    aaa_ch3 = u_sat_ch3[i]
    aaa_ch4 = u_sat_ch4[i]
    aaa_ch5 = u_sat_ch5[i]
    aaa_ch6 = u_sat_ch6[i]
    aaa_ch7 = u_sat_ch7[i]
    aaa_ch8 = u_sat_ch8[i]
    if aaa_ch1 > A_max:
        aaa_ch1 = A_max
    elif aaa_ch1 < min_val:
        aaa_ch1 = min_val

    if aaa_ch2 > A_max:
        aaa_ch2 = A_max
    elif aaa_ch2 < min_val:
        aaa_ch2 = min_val

    if aaa_ch3 > A_max:
        aaa_ch3 = A_max
    elif aaa_ch3 < min_val:
        aaa_ch3 = min_val

    if aaa_ch4 > A_max:
        aaa_ch4 = A_max
    elif aaa_ch4 < min_val:
        aaa_ch4 = min_val

    if aaa_ch5 > A_max:
        aaa_ch5 = A_max
    elif aaa_ch5 < min_val:
        aaa_ch5 = min_val

    if aaa_ch6 > A_max:
        aaa_ch6 = A_max
    elif aaa_ch6 < min_val:
        aaa_ch6 = min_val

    if aaa_ch7 > A_max:
        aaa_ch7 = A_max
    elif aaa_ch7 < min_val:
        aaa_ch7 = min_val

    if aaa_ch8 > A_max:
        aaa_ch8 = A_max
    elif aaa_ch8 < min_val:
        aaa_ch8 = min_val

    u_app_ch1[i] = aaa_ch1
    u_app_ch2[i] = aaa_ch2
    u_app_ch3[i] = aaa_ch3
    u_app_ch4[i] = aaa_ch4
    u_app_ch5[i] = aaa_ch5
    u_app_ch6[i] = aaa_ch6
    u_app_ch7[i] = aaa_ch7
    u_app_ch8[i] = aaa_ch8
    time_table = 5
    v1_table = aaa_ch1
    v2_table = aaa_ch2
    v3_table = aaa_ch3
    v4_table = aaa_ch4
    v5_table = aaa_ch5
    v6_table = aaa_ch6
    v7_table = aaa_ch7
    v8_table = aaa_ch8

    # field names
    csv_table = {'t(s)': [time_table], 'V1(v)': [v1_table], 'V2(v)': [v2_table], 'V3(v)': [v3_table],
                 'V4(v)': [v4_table], 'V5(v)': [v5_table], 'V6(v)': [v6_table], 'V7(v)': [v7_table],
                 'V8(v)': [v8_table]}
    print(dataFrame)
    dataFrame = pd.DataFrame(csv_table)
    dataFrame.to_csv('~/Desktop/Close_Loop_Actuation/Wound_1.csv', mode='a', index=False, header=False)

    # plotting the points

    # naming the x axis
    if time.time() - start_time > 3600:
        plt.scatter(time_vector_hour[:i + 1], i_1_value[:i + 1])
        plt.scatter(time_vector_hour[:i + 1], i_2_value[:i + 1])
        plt.scatter(time_vector_hour[:i + 1], i_3_value[:i + 1])
        plt.scatter(time_vector_hour[:i + 1], i_4_value[:i + 1])
        plt.scatter(time_vector_hour[:i + 1], i_5_value[:i + 1])
        plt.scatter(time_vector_hour[:i + 1], i_6_value[:i + 1])
        plt.scatter(time_vector_hour[:i + 1], i_7_value[:i + 1])
        plt.xlabel('time (hours)')

    elif time.time() - start_time > 60:
        plt.scatter(time_vector_min[:i + 1], i_1_value[:i + 1])
        plt.scatter(time_vector_min[:i + 1], i_2_value[:i + 1])
        plt.scatter(time_vector_min[:i + 1], i_3_value[:i + 1])
        plt.scatter(time_vector_min[:i + 1], i_4_value[:i + 1])
        plt.scatter(time_vector_min[:i + 1], i_5_value[:i + 1])
        plt.scatter(time_vector_min[:i + 1], i_6_value[:i + 1])
        plt.scatter(time_vector_min[:i + 1], i_7_value[:i + 1])
        plt.scatter(time_vector_min[:i + 1], i_8_value[:i + 1])
        plt.xlabel('time (mins)')
    else:
        plt.scatter(time_vector[:i + 1], i_1_value[:i + 1])
        plt.scatter(time_vector[:i + 1], i_2_value[:i + 1])
        plt.scatter(time_vector[:i + 1], i_3_value[:i + 1])
        plt.scatter(time_vector[:i + 1], i_4_value[:i + 1])
        plt.scatter(time_vector[:i + 1], i_5_value[:i + 1])
        plt.scatter(time_vector[:i + 1], i_6_value[:i + 1])
        plt.scatter(time_vector[:i + 1], i_7_value[:i + 1])
        plt.scatter(time_vector[:i + 1], i_8_value[:i + 1])
        plt.xlabel('time (seconds)')

    # naming the y axis
    plt.ylabel('current (uA)')

    # giving a title to my graph
    plt.title('Current (sampled every 5 second)')

    # function to show the plot
    plt.title("Wound 1")
    plt.show(block=False)

plt.show()



















