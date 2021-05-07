import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from generate_patient_fill_data import MedFillData, MedHistory
from medication_processing_dicts import mapped_time_invar_names, mult_meds_to_single_meds, str_units_conversions, nice_strength_units_conversion

def makeFiddleEntry(patid, time, var, value):
    entry = {}
    
    entry['ID'] = patid
    entry['time'] = time
    entry['var'] = var
    entry['value'] = value
    
    return entry

#################### GETTING THE TIME INDEPENDENT DATA IN FIDDLE FORM ###########################
### Drop features from the df that are not usable
def drop_features(df):
    df = df.drop(['eligeff', 'eligend', 'conf_id', 'fst_surg', 'proc_desc', 'assigned_prescriber_npi',
              'initial_postop_rx_fill_dt', 'refill_end_dt', 'surg_dt', 'fst_serv_dt', 'last_fill_rx_dt', 'fst_serv_dt',
              'lst_serv_dt', 'flg_refill','dstatus', 'los', 'days_post', 'follow_ge24', 'ane_181_365', 'periop_opioid', 
              'periop_opioid_adj', 'p0_90_opioid', 'p14_90_opioid', 'p91_180_opioid', 'p181_365_opioid', 'ccs_650_post', 
              'ccs_651_post', 'ccs_652_post', 'ccs_656_post', 'ccs_657_post', 'ccs_658_post', 'ccs_659_post', 'ccs_660_post', 
              'ccs_661_post', 'ccs_662_post', 'ccs_663_post', 'ccs_670_post', 'ccs_84_post', 'ccs_202_post', 'ccs_203_post', 
              'ccs_204_post', 'ccs_205_post', 'ccs_251_post', 'depress_post', 'anxiety_post', 'migraine_post', 'tmjd_post', 
              'back_post', 'neck_post', 'tension_post', 'headpain_post', 'neuralgia_post', 'dryeyes_post', 'fibromyalgia_post', 
              'chestpain_post', 'esophagus_post', 'bowel_post', 'cystitis_post', 'vulvo_post', 'endomet_post', 'arthritis_post', 
              'dyspepsia_post', 'sicca_post', 'tinnitus_post', 'fatigue_post', 'insomnia_post', 'flg_initial_postop_rx', 
              'tot_initial_postop_rx_ome', 'num_initial_postop_script', 'initial_postop_rx_fill_dt', 'refill_end_dt'],1)
    #drop the features Vidhya recommended I drop
    df = df.drop(['yrdob', 'ane_prior_180', 'ane_for_surg', 'prior_365_31_opioid', 'depress', 'anxiety', 'flg_preop_rx', 
                  'total_preop_rx_ome', 'admit_month', 'last_fill_month', 'last_fill_month', 'last_fill_month', 
                  'longest_fill_streak', 'fill_recency'],1)
    # drop opioid features
    df = df.drop(['outcome_status',
                 'total_preop_rx_ome_0',
                 'total_preop_rx_ome_150',
                 'total_preop_rx_ome_300',
                 'total_preop_rx_ome_850',
                 'total_preop_rx_ome_10000',
                 'num_months_rx_filled_0',
                 'num_months_rx_filled_1',
                 'num_months_rx_filled_3',
                 'num_months_rx_filled_13',
                 'fill_recency_0',
                 'fill_recency_1',
                 'fill_recency_2',
                 'fill_recency_5',
                 'fill_recency_12',
                 'longest_fill_streak_0',
                 'longest_fill_streak_1',
                 'longest_fill_streak_2',
                 'longest_fill_streak_13',
                 'flg_prolonged_use',
                 'num_preop_rx',
                 'num_months_rx_filled',], 1)
    df[['tobacco', 'ccs_651', 'ccs_652', 'ccs_656', 'ccs_657','ccs_658','ccs_659','ccs_660', 
         'ccs_661','ccs_662','ccs_663','ccs_670', 'ccs_84','ccs_202','ccs_203','ccs_204','ccs_205','ccs_251', 
         'migraine','tmjd','back','neck','tension','headpain','neuralgia','dryeyes','fibromyalgia','chestpain',
         'esophagus','bowel','cystitis','vulvo','endomet','arthritis','dyspepsia','sicca','tinnitus','fatigue',
         'insomnia']].fillna(value=0, inplace=True)
    return df

if __name__ == '__main__':

    # get path from argument
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--demographic_dataframes_path', type=str, required=True)
    parser.add_argument('--patient_fill_info_path', type=str, required=True)
    parser.add_argument('--target_path', type=str, required=True)
    args = parser.parse_args()
    dataframe_path = args.demographic_dataframes_path
    patient_fill_info_path = args.patient_fill_info_path
    target_path = args.target_path

    ### Import dictionary with patient fill information
    with open(patient_fill_info_path + 'patient_fill_data_imp_meds.pickle', 'rb') as f:
        pat_fill_data = pickle.load(f)

    ### Import demographic data (pre-separated)
    naive_train = pd.read_pickle(dataframe_path + 'naive_train.pkl')
    naive_test = pd.read_pickle(dataframe_path + 'naive_test.pkl')

    #################### GETTING THE TIME DEPENDENT DATA IN FIDDLE FORM ###########################
    ### Just fill in the patient medication data
    fiddle_data = []
    for patid in naive_train.patid or patid in naive_test.patid:
        med_hist = pat_fill_data_imp_med[patid]
        
        for med_fill in med_hist.med_list:
            time = 365 - med_fill.fill_dt
            
            fiddle_data.append(makeFiddleEntry(patid, time, med_fill.name + '_quantity', med_fill.quantity))
            
            if med_fill.name in mult_meds_to_single_meds and med_fill.str_unit == '' and med_fill.strength_str != '':
                meds = mult_meds_to_single_meds[med_fill.name]
                strs = str_units_conversions[med_fill.name][med_fill.strength_str]
                if len(meds) != len(strs):
                    print(med_fill.name + " has error")
                for med, strength in zip(meds, strs):
                    amount = strength * med_fill.quantity
                    fiddle_data.append(makeFiddleEntry(patid, time, med + '_amount', amount))
            else:
                strength_multiple = 1
                if med_fill.str_unit != '':
                    strength_multiple = med_fill.strength * nice_strength_units_conversion[med_fill.str_unit]
                elif med_fill.strength_str != '':
                    strength_multiple = str_units_conversions[med_fill.name][med_fill.strength_str][0]
                amount = strength_multiple * med_fill.quantity
                fiddle_data.append(makeFiddleEntry(patid, time, med + '_amount', amount))  

    ### This will get rid of duplicates
    ### First, identify the first of each of the entries with matching (ID, time, var)
    first_of_its_kind = {}

    for i in range(len(fiddle_data)):
        tuple_id = (fiddle_data[i]['ID'], fiddle_data[i]['time'], fiddle_data[i]['var'])
        if tuple_id not in first_of_its_kind:
            first_of_its_kind[tuple_id] = i

    ### Sum up all of the values and store in the first of its kind.
    for i in range(len(fiddle_data)):
        tuple_id = (fiddle_data[i]['ID'], fiddle_data[i]['time'], fiddle_data[i]['var'])
        first_index = first_of_its_kind[tuple_id]
        if i != first_index:
            fiddle_data[first_index]['value'] += fiddle_data[i]['value']

    ### Form FIDDLE dataframe
    fiddle_df = pd.DataFrame(fiddle_data)

    ### Rename to fit FIDDLE format
    fiddle_df = fiddle_df.rename(columns={'time':'t', 'var':'variable_name', 'value':'variable_value'})

    ### Reorder to fit FIDDLE format
    cols = ['ID', 't', 'variable_name', 'variable_value']
    fiddle_df = fiddle_df[cols]

    ### Drop all duplicate rows, keep the first one where we stored the sum.
    fiddle_df.drop_duplicates(subset=['ID', 't', 'variable_name'], keep='first', inplace=True)

    ### Make the df, drop features, and fill NAs with 0s
    df = pd.concat([naive_train, naive_test], axis=0)
    df = drop_features(df)
    df = df.fillna(0)

    ### Regularize the binary features.
    from pandas.api.types import is_numeric_dtype
    from pandas.api.types import is_string_dtype

    def is_binary(series):
        return len(series.unique()) <= 2 and not is_string_dtype(series)

    bool_numeric_cols = [col for col in df if is_binary(df[col])]

    print(bool_numeric_cols)

    bool_to_str = {1 : 'y', 
                  '1' : 'y',
                  1.0 : 'y',
                  0 : 'n', 
                  '0' : 'n',
                  0.0 : 'n'}

    for col in list(df):
        if col in bool_numeric_cols:
            print(col)
            df[col] = df.apply(lambda row: bool_to_str[row[col]], axis=1)

    ### Rename column names to more readable ones
    df = df.rename(columns=mapped_names)

    ### Get the list of time invariant features
    time_invar_features = list(df)
    time_invar_features.remove('patid')

    ### Collect the time invariant data from the dataframe
    fiddle_data_time_invar = []
    for index, row in df.iterrows():
        if row.patid in naive_train.patid.values or row.patid in naive_test.patid.values:
            for feature in time_invar_features:
                fiddle_data_time_invar.append(makeFiddleEntry(row.patid, np.nan, feature, row[feature]))
    fiddle_df_time_invar = pd.DataFrame(fiddle_data_time_invar)

    ### Rename column names to fiddle format
    fiddle_df_time_invar = fiddle_df_time_invar.rename(columns={'time':'t', 'var':'variable_name', 'value':'variable_value'})
    cols = ['ID', 't', 'variable_name', 'variable_value']
    fiddle_df_time_invar = fiddle_df_time_invar[cols]

    ### Drop duplicates
    fiddle_df_time_invar.drop_duplicates(subset=['ID', 't', 'variable_name'], keep='first', inplace=True)

    #################### JOIN FOR COMPLETE FIDDLE FORMAT DF ###########################
    fiddle_complete = pd.concat([fiddle_df, fiddle_df_time_invar], ignore_index=True)

    ### Save
    fiddle_complete.to_csv(target_path + 'fiddle_format_data.csv', index=False)

    #################### GET POPULATION DF (SORTED BY PATID) ###########################
    patids = naive_train.patid.values.tolist() + naive_test.patid.values.tolist()
    patids.sort()
    pop_df = pd.DataFrame(patids, columns=['ID'])

    ### Save
    pop_df.to_csv(target_path + 'fiddle_format_pop.csv', index=False)