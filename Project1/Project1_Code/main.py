import pandas as pd
import numpy as np

# Reading the CGMData.csv and InsulinData.csv

cgmData = pd.read_csv('CGMData.csv',low_memory = False, usecols = ['Date','Time','Sensor Glucose (mg/dL)'])
# cgmData = pd.read_csv('/Users/utkarshsharma/Documents/CSE572_DM/Project1/Project1StudentFiles/CGMData.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])


insulinData = pd.read_csv('InsulinData.csv',low_memory = False)
# insulinData = pd.read_csv('/Users/utkarshsharma/Documents/CSE572_DM/Project1/Project1StudentFiles/InsulinData.csv',low_memory=False)

# Deriving the new feature Date + TimeStamp for CGMData

cgmData['date_time_stamp'] = pd.to_datetime(cgmData['Date'] + ' ' + cgmData['Time'])

# Finding out the unique dates to be removed

remove_dates = cgmData[cgmData['Sensor Glucose (mg/dL)'].isna()]['Date'].unique()

cgmData = cgmData.set_index('Date').drop(index=remove_dates).reset_index()

# Deriving the new feature Date + TimeStamp for InsulinData

insulinData['date_time_stamp'] = pd.to_datetime(insulinData['Date'] + ' ' + insulinData['Time'])

# Finding out the start of Auto mode

automode_start = insulinData.sort_values(by = 'date_time_stamp', ascending = True).loc[insulinData['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF'].iloc[0]['date_time_stamp']

automode_data_df = cgmData.sort_values(by = 'date_time_stamp', ascending = True).loc[cgmData['date_time_stamp'] >= automode_start]

# Finding out the start of Manual mode

manualmode_data_df = cgmData.sort_values(by = 'date_time_stamp',ascending = True).loc[cgmData['date_time_stamp'] < automode_start]

automode_data_df_dateIndex = automode_data_df.set_index('date_time_stamp')

array1 = automode_data_df_dateIndex.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(lambda x: x > 0.7 * 288).dropna().index.tolist()

automode_data_df_dateIndex = automode_data_df_dateIndex.loc[automode_data_df_dateIndex['Date'].isin(array1)]

# Value Calculations for Auto Mode

# % in Hyperglycemia (> 180 mg/dL) - Daytime, Overnight, Wholeday

percentage_time_in_hyperglycemia_daytime_automode = (automode_data_df_dateIndex.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[automode_data_df_dateIndex['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hyperglycemia_overnight_automode = (automode_data_df_dateIndex.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[automode_data_df_dateIndex['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hyperglycemia_wholeday_automode = (automode_data_df_dateIndex.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[automode_data_df_dateIndex['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# % in Hyperglycemia critical (> 250 mg/dL) -  Daytime, Overnight, Wholeday

percentage_time_in_hyperglycemia_critical_daytime_automode = (automode_data_df_dateIndex.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[automode_data_df_dateIndex['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hyperglycemia_critical_overnight_automode = (automode_data_df_dateIndex.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[automode_data_df_dateIndex['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hyperglycemia_critical_wholeday_automode = (automode_data_df_dateIndex.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[automode_data_df_dateIndex['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# %  in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL) - Daytime, Overnight, Wholeday

percentage_time_in_range_daytime_automode = (automode_data_df_dateIndex.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(automode_data_df_dateIndex['Sensor Glucose (mg/dL)']>=70) & (automode_data_df_dateIndex['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_range_overnight_automode = (automode_data_df_dateIndex.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(automode_data_df_dateIndex['Sensor Glucose (mg/dL)']>=70) & (automode_data_df_dateIndex['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_range_wholeday_automode = (automode_data_df_dateIndex.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(automode_data_df_dateIndex['Sensor Glucose (mg/dL)']>=70) & (automode_data_df_dateIndex['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


#  %  in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL) -  Daytime, Overnight, Wholeday

percentage_time_in_range_sec_daytime_automode = (automode_data_df_dateIndex.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(automode_data_df_dateIndex['Sensor Glucose (mg/dL)']>=70) & (automode_data_df_dateIndex['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_range_sec_overnight_automode = (automode_data_df_dateIndex.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(automode_data_df_dateIndex['Sensor Glucose (mg/dL)']>=70) & (automode_data_df_dateIndex['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_range_sec_wholeday_automode = (automode_data_df_dateIndex.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(automode_data_df_dateIndex['Sensor Glucose (mg/dL)']>=70) & (automode_data_df_dateIndex['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


#  % in hypoglycemia level 1 (CGM < 70 mg/dL) -  Daytime, Overnight, Wholeday

percentage_time_in_hypoglycemia_lv1_daytime_automode = (automode_data_df_dateIndex.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[automode_data_df_dateIndex['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hypoglycemia_lv1_overnight_automode = (automode_data_df_dateIndex.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[automode_data_df_dateIndex['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hypoglycemia_lv1_wholeday_automode = (automode_data_df_dateIndex.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[automode_data_df_dateIndex['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

# % in hypoglycemia level 2 (CGM < 54 mg/dL) -  Daytime, Overnight, Wholeday

percentage_time_in_hypoglycemia_lv2_daytime_automode = (automode_data_df_dateIndex.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[automode_data_df_dateIndex['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hypoglycemia_lv2_overnight_automode = (automode_data_df_dateIndex.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[automode_data_df_dateIndex['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hypoglycemia_lv2_wholeday_automode = (automode_data_df_dateIndex.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[automode_data_df_dateIndex['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

# Value Calculations for Manual Mode

manualmode_data_df_index = manualmode_data_df.set_index('date_time_stamp')

array2 = manualmode_data_df_index.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(lambda x:x > 0.7 * 288).dropna().index.tolist()

manualmode_data_df_index = manualmode_data_df_index.loc[manualmode_data_df_index['Date'].isin(array2)]


#  % in Hyperglycemia (> 180 mg/dL) -  Daytime, Overnight, Wholeday

percentage_time_in_hyperglycemia_daytime_manualmode = (manualmode_data_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manualmode_data_df_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hyperglycemia_overnight_manualmode = (manualmode_data_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manualmode_data_df_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hyperglycemia_wholeday_manualmode = (manualmode_data_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manualmode_data_df_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# % in Hyperglycemia critical (> 250 mg/dL) - Daytime, Overnight, Wholeday

percentage_time_in_hyperglycemia_critical_daytime_manualmode = (manualmode_data_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manualmode_data_df_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hyperglycemia_critical_overnight_manualmode = (manualmode_data_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manualmode_data_df_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hyperglycemia_critical_wholeday_manualmode = (manualmode_data_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manualmode_data_df_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# %  in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL) -  Daytime, Overnight, Wholeday

percentage_time_in_range_daytime_manualmode = (manualmode_data_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manualmode_data_df_index['Sensor Glucose (mg/dL)']>=70) & (manualmode_data_df_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_range_overnight_manualmode = (manualmode_data_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manualmode_data_df_index['Sensor Glucose (mg/dL)']>=70) & (manualmode_data_df_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_range_wholeday_manualmode = (manualmode_data_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manualmode_data_df_index['Sensor Glucose (mg/dL)']>=70) & (manualmode_data_df_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# %  in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL) - Daytime, Overnight, Wholeday

percentage_time_in_range_sec_daytime_manualmode = (manualmode_data_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manualmode_data_df_index['Sensor Glucose (mg/dL)']>=70) & (manualmode_data_df_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_range_sec_overnight_manualmode = (manualmode_data_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manualmode_data_df_index['Sensor Glucose (mg/dL)']>=70) & (manualmode_data_df_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_range_sec_wholeday_manualmode = (manualmode_data_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(manualmode_data_df_index['Sensor Glucose (mg/dL)']>=70) & (manualmode_data_df_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


#  % in hypoglycemia level 1 (CGM < 70 mg/dL) -  Daytime, Overnight, Wholeday

percentage_time_in_hypoglycemia_lv1_daytime_manualmode = (manualmode_data_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manualmode_data_df_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hypoglycemia_lv1_overnight_manualmode = (manualmode_data_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manualmode_data_df_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hypoglycemia_lv1_wholeday_manualmode = (manualmode_data_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manualmode_data_df_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# % in hypoglycemia level 2 (CGM < 54 mg/dL) -  Daytime, Overnight, Wholeday

percentage_time_in_hypoglycemia_lv2_daytime_manualmode = (manualmode_data_df_index.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manualmode_data_df_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hypoglycemia_lv2_overnight_manualmode = (manualmode_data_df_index.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manualmode_data_df_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percentage_time_in_hypoglycemia_lv2_wholeday_manualmode = (manualmode_data_df_index.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[manualmode_data_df_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


# Calculating the count of Manual and Automode days to be used for mean value calculation

count_manualmode_days = len(array2)

count_automode_days = len(array1)

# Setting up the header names in the order as specified in the Results.csv

headerNames = ['Modes',             
              'Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)',
              'Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)',
              'Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
              'Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)',
              'Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)',
              'Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)',
              'Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)',
              'Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)',
              'Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
              'Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)',
              'Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)',
              'Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)',
              'Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)',
              'Whole day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)',
              'Whole day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
              'Whole day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)',
              'Whole day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)',
              'Whole day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)']

# Setting up the dummy values of 1.1 for both types of modes making the matrix of size 2 X 19

dummy = [np.float64(1.1), np.float64(1.1)]
results_df = pd.DataFrame(columns = headerNames)
results_df['Modes'] = ['Manual Mode','Auto Mode']


results_df['Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)'] = [round((sum(percentage_time_in_hyperglycemia_overnight_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_hyperglycemia_overnight_automode)/count_automode_days), 4)]
results_df['Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)'] = [round((sum(percentage_time_in_hyperglycemia_critical_overnight_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_hyperglycemia_critical_overnight_automode)/count_automode_days),4)]
results_df['Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)'] = [round((sum(percentage_time_in_range_overnight_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_range_overnight_automode)/count_automode_days), 4)]
results_df['Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)'] = [round((sum(percentage_time_in_range_sec_overnight_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_range_sec_overnight_automode)/count_automode_days), 4)]
results_df['Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)'] = [round((sum(percentage_time_in_hypoglycemia_lv1_overnight_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_hypoglycemia_lv1_overnight_automode)/count_automode_days), 4)]
results_df['Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] = [round((sum(percentage_time_in_hypoglycemia_lv2_overnight_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_hypoglycemia_lv2_overnight_automode)/count_automode_days), 4)]


results_df['Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)'] = [round((sum(percentage_time_in_hyperglycemia_daytime_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_hyperglycemia_daytime_automode)/count_automode_days), 4)]
results_df['Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)'] = [round((sum(percentage_time_in_hyperglycemia_critical_daytime_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_hyperglycemia_critical_daytime_automode)/count_automode_days), 4)]
results_df['Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)'] = [round((sum(percentage_time_in_range_daytime_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_range_daytime_automode)/count_automode_days), 4)]
results_df['Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)'] = [round((sum(percentage_time_in_range_sec_daytime_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_range_sec_daytime_automode)/count_automode_days), 4)]
results_df['Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)'] = [round((sum(percentage_time_in_hypoglycemia_lv1_daytime_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_hypoglycemia_lv1_daytime_automode)/count_automode_days), 4)]
results_df['Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] = [round((sum(percentage_time_in_hypoglycemia_lv2_daytime_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_hypoglycemia_lv2_daytime_automode)/count_automode_days), 4)]


results_df['Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)'] = [round((sum(percentage_time_in_hyperglycemia_wholeday_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_hyperglycemia_wholeday_automode)/count_automode_days), 4)]
results_df['Whole day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)'] = [round((sum(percentage_time_in_hyperglycemia_critical_wholeday_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_hyperglycemia_critical_wholeday_automode)/count_automode_days), 4)]
results_df['Whole day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)'] = [round((sum(percentage_time_in_range_wholeday_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_range_wholeday_automode)/count_automode_days), 4)]
results_df['Whole day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)'] = [round((sum(percentage_time_in_range_sec_wholeday_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_range_sec_wholeday_automode)/count_automode_days), 4)]
results_df['Whole day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)'] = [round((sum(percentage_time_in_hypoglycemia_lv1_wholeday_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_hypoglycemia_lv1_wholeday_automode)/count_automode_days), 4)]
results_df['Whole day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] = [round((sum(percentage_time_in_hypoglycemia_lv2_wholeday_manualmode)/count_manualmode_days), 4), round((sum(percentage_time_in_hypoglycemia_lv2_wholeday_automode)/count_automode_days), 4)]


results_df = pd.concat([results_df,pd.DataFrame(dummy)], axis = 1)
results_df.set_index('Modes', inplace = True)
results_df.to_csv('Results.csv', index = False, header = None)
# results_df.to_csv('/Users/utkarshsharma/Documents/CSE572_DM/Project1/Project1StudentFiles/Results.csv',index = False, header = None)
