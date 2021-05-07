import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import sklearn as sk
import numpy as np
from matplotlib import pyplot as plt
import pickle
from collections import OrderedDict, defaultdict
import medication_processing_dicts

## Preprocessing of dataset
def process_dataset(df, include_naive):
    ## List of postoperative features to drop, as well as redundant features.
    df = df.drop(['patid', 'eligeff', 'eligend', 'division', 'conf_id', 'fst_surg', 'proc_desc', 'assigned_prescriber_npi',
              'initial_postop_rx_fill_dt', 'refill_end_dt', 'surg_dt', 'fst_serv_dt', 'last_fill_rx_dt', 'fst_serv_dt',
              'lst_serv_dt', 'flg_refill','dstatus', 'los', 'days_post', 'follow_ge24', 'ane_181_365', 'periop_opioid', 
              'periop_opioid_adj', 'p0_90_opioid', 'p14_90_opioid', 'p91_180_opioid', 'p181_365_opioid', 'ccs_650_post', 
              'ccs_651_post', 'ccs_652_post', 'ccs_656_post', 'ccs_657_post', 'ccs_658_post', 'ccs_659_post', 'ccs_660_post', 
              'ccs_661_post', 'ccs_662_post', 'ccs_663_post', 'ccs_670_post', 'ccs_84_post', 'ccs_202_post', 'ccs_203_post', 
              'ccs_204_post', 'ccs_205_post', 'ccs_251_post', 'depress_post', 'anxiety_post', 'migraine_post', 'tmjd_post', 
              'back_post', 'neck_post', 'tension_post', 'headpain_post', 'neuralgia_post', 'dryeyes_post', 'fibromyalgia_post', 
              'chestpain_post', 'esophagus_post', 'bowel_post', 'cystitis_post', 'vulvo_post', 'endomet_post', 'arthritis_post', 
              'dyspepsia_post', 'sicca_post', 'tinnitus_post', 'fatigue_post', 'insomnia_post', 'flg_initial_postop_rx', 
              'tot_initial_postop_rx_ome', 'num_initial_postop_script', 'initial_postop_rx_fill_dt', 'refill_end_dt', 'yrdob', 'ane_prior_180', 'ane_for_surg', 'prior_365_31_opioid', 'depress', 'anxiety', 'flg_preop_rx', 
              'total_preop_rx_ome', 'admit_month', 'last_fill_month', 'last_fill_month', 'last_fill_month', 
              'longest_fill_streak', 'fill_recency'],1)
    ## Fill in 0s for NA values (supposed to be 0's)
    df[['tobacco', 'ccs_651', 'ccs_652', 'ccs_656', 'ccs_657','ccs_658','ccs_659','ccs_660', 
         'ccs_661','ccs_662','ccs_663','ccs_670', 'ccs_84','ccs_202','ccs_203','ccs_204','ccs_205','ccs_251', 
         'migraine','tmjd','back','neck','tension','headpain','neuralgia','dryeyes','fibromyalgia','chestpain',
         'esophagus','bowel','cystitis','vulvo','endomet','arthritis','dyspepsia','sicca','tinnitus','fatigue',
         'insomnia']].fillna(value=0, inplace=True)
    def label_col(row, col, answer):
        if row[col] == answer:
            return 1.0
        else:
            return 0.0

    def is_binary_or_numeric(series, allow_na=False):
        if allow_na:
            series.dropna(inplace=True)
        return sorted(series.unique()) == [0, 1] or sorted(series.unique()) == ['0', '1'] or sorted(series.unique()) == [0.0, 1.0] or is_numeric_dtype(series)

    bool_numeric_cols = [col for col in df if is_binary_or_numeric(df[col], allow_na=True)]
    for col in list(df):
        if col not in bool_numeric_cols:
            for answer in df[col].unique():
                df[col+str(answer)] = df.apply(lambda row: label_col(row,col,answer), axis=1)
            df = df.drop(col, axis=1)
    return df

class MedHistory:
    def __init__(self):
        self.med_list = []
        self.op_list = []

class MedFillData:
    def __init__(self, name, ndc, fill_dt, strength, strength_str, quantity, volume = -1, str_unit='', vol_unit='', is_opioid = 0):
        self.name = name
        self.ndc = ndc
        self.fill_dt = fill_dt
        self.strength = strength
        self.strength_str = strength_str
        self.quantity = quantity
        self.volume = volume
        self.str_unit = str_unit
        self.vol_unit = vol_unit
        self.is_opioid = is_opioid

## Engineers features from features including demographics and comorbidities (df_dc),
## and other medication use (df_pharma), with df_ndc providing specific information about each
## drug
def engineer_pharma_data_combined(df_dc, df_pharma, df_ndc):
    patient_fill_data = {}
    for index, row in df_dc.iterrows():
        
        patient_fill_data[row.patid] = MedHistory()
        
        for _, row_ in df_pharma.loc[df_pharma.Patid == row.patid].iterrows():
            if type(row.fst_serv_dt) == pd._libs.tslibs.timestamps.Timestamp and type(row_.Fill_Dt) == pd._libs.tslibs.timestamps.Timestamp:
                drug = row_.Gnrc_Nm
                if row_.Fill_Dt < row.fst_serv_dt:
                    fill_dt = pd.Timedelta(row.fst_serv_dt - row_.Fill_Dt).days
                    
                    df_ndc_sub = df_ndc[df_ndc.ndc == row_.Ndc]
                    name = drug
                    ndc = row_.Ndc
                    strength = df_ndc_sub.drg_strgth_nbr.unique()[0]
                    strength_str = df_ndc[df_ndc.ndc == ndc].drg_strgth_desc.unique()[0]
                    str_unit = df_ndc_sub.drg_strgth_unit_desc.unique()[0].lower()
                    quantity = row_.Quantity
                    volume = df_ndc_sub.drg_strgth_vol_nbr.unique()[0]
                    vol_unit = df_ndc_sub.drg_strgth_vol_unit_desc.unique()[0].lower()
                    
                    med_fill = MedFillData(name, ndc, fill_dt, strength, strength_str, quantity, volume, str_unit, vol_unit, row_.OPIOID)
                    patient_fill_data[row.patid].med_list.append(med_fill)
                            
        if index % 10000 == 0:
            print("Gen med data: at patient index " + str(index))

    print("saving dict...")
    with open('/home/jaewonh/data/patient_fill_data.pickle', 'wb') as handle:
        pickle.dump(patient_fill_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return patient_fill_data

if __name__ == '__main__':

    # get path from argument
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataframe_path', type=str, required=True)
    parser.add_argument('--target_path', type=str, required=True)
    args = parser.parse_args()
    dataframe_path = args.dataframe_path
    target_path = args.target_path

    # We only wish to count the meds which 0.1%<, <99.9% of patients used.
    important_med_list = []

    # maps each medication to how many patients filled a prescription for it.
    num_patients_per_med = defaultdict(int)

    df_dc = pd.read_pickle(dataframe_path + 'gen_data.pkl')
    df_ndc = pd.read_stata(dataframe_path + 'ndc_optum.dta')
    df_pharma = pd.read_pickle(dataframe_path + 'pharma_data.pkl')

    patient_fill_data = engineer_pharma_data_combined(df_dc, df_pharma, df_ndc)

    ### After obtaining the dictionary, we can get a list of meds that more than 0.1%
    ### and less than 99.9% of patients used.
    for patid, medhist in patient_fill_data.items():
        for med_fill in medhist.med_list:
            num_patients_per_med[med_fill.name] += 1

    for med, count in num_patients_per_med.items():
        if count > 0.001*90318 and count < 0.999*90318:
            important_med_list.append(med)

    ### get a dictionary with only the important meds
    patient_fill_data_imp_meds = {}
    for patid, med_history in patient_fill_data.items():
        patient_fill_data_imp_meds[patid] = MedHistory()
        for med in med_history.med_list:
            if med.name in important_med_list:
                patient_fill_data_imp_meds[patid].med_list.append(med)

    with open(target_path + 'patient_fill_data_imp_meds.pickle', 'wb') as handle:
        pickle.dump(patient_fill_data_imp_meds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(target_path + 'num_patients_per_med.pickle', 'wb') as handle:
        pickle.dump(num_patients_per_med, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(target_path + 'important_meds.pickle', 'wb') as handle:
        pickle.dump(important_med_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

