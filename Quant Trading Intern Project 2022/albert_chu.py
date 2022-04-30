import pandas as pd
import numpy as np
import glob
import csv

all_files = glob.glob("Data/*.csv")
df_list = ['ADB', 'ADS', 'BPB', 'BPS', 'CDB', 'CDS', 'ECB', 'ECS', 'JYB', 'JYS', 'SFB', 'SFS']
name_list = ['ADB', 'ADS', 'BPB', 'BPS', 'CDB', 'CDS', 'ECB', 'ECS', 'JYB', 'JYS', 'SFB', 'SFS']

#read into corresponding dfs 
for i, csv in enumerate(all_files):
    df_list[i] = pd.read_csv(csv)
    name = name_list[i]
    df_list[i].columns = [name+'Date',name+'Open',name+'High',name+'Low',name+'Close',name+'Volume',name+'Open Interest']
    df_list[i].drop(columns=[name+'Open',name+'High',name+'Low',], inplace=True)
    df_list[i].drop(columns=[name+'Volume',name+'Open Interest'], inplace=True)
    df_list[i].set_index(name+'Date', inplace=True)
    df_list[i].rename_axis("Date", inplace=True)

specs = pd.read_csv("contract_specs.csv")
specs.set_index("Ticker", inplace=True)

total = df_list[0].copy()
for i in range(1, len(df_list)):
    total = pd.merge(total, df_list[i], on='Date', how='outer', sort=True).fillna(method='ffill')

def trade_logic(df_raw, specs=specs, lookback= 252, num_stdevs = 0.5, market_list= ['AD', 'BP', 'CD', 'EC', 'JY', 'SF']):
    #df, specs, look_back, num_stdevs, min_days_held, market_list
    name_list = ['ADB', 'ADS', 'BPB', 'BPS', 'CDB', 'CDS', 'ECB', 'ECS', 'JYB', 'JYS', 'SFB', 'SFS']
    real_list = ['AD', 'BP', 'CD', 'EC', 'JY', 'SF']

    start = 20070102
    
    df_start = df_raw.index.get_loc(start)
    i_start = df_start - lookback 
    df = df_raw.iloc[i_start:, :].copy()
    
    transac_cost = {}
    
    df['Sizing Sum'] = 0
    
    for market in market_list:
        mult_i = specs.loc[market, "Multiplier"]
        df[market+' Price'] = df[market+'BClose']
        
        df[market+' Returns'] = (df[market+'BClose'] - (df[market+'BClose'].shift(1))) / (df[market+'SClose'].shift(1))

        df[market+' Vol'] = df[market+' Returns'].rolling(lookback).std()
        df[market+' Pct Change'] = (df[market+'BClose'] - (df[market+'BClose'].shift(lookback))) / (df[market+'SClose'].shift(lookback))
        df[market+' Trend'] = df[market+' Pct Change'] / df[market+' Vol']
        
        df[market+' P&L'] = 0
        df[market+' Position'] = 0
        
        df[market+' Pullback Sign'] = (np.sign(df[market+' Returns']) != np.sign(df[market+' Trend'])).shift(1)
        df[market+' Pullback Thld'] = (np.abs(df[market+' Returns'] - df[market+' Trend']) >= np.abs(num_stdevs * df[market+' Vol'])).shift(1)
    
        # Append Transaction Cost to Dictionary 
        tick_i = specs.loc[market, 'Tick Size']
        exch_i = specs.loc[market, 'Exchange Fees']
        comm_i = specs.loc[market, 'Commission']
        nfa_i = specs.loc[market, 'NFA Fees']
        
        transac_cost[market] = (tick_i * mult_i) + exch_i + comm_i + nfa_i
        
        df['Sizing Sum'] += (1/(df[market+' Vol'] * mult_i))
    
    for market in market_list:
        pos_sign = np.sign(df[market+' Trend'])
        df[market+' Theo Position Size'] = pos_sign * 10 * np.abs((1/(df[market+' Vol'] * specs.loc[market, "Multiplier"])) / df['Sizing Sum'])
#         print(df[market+' Theo Position Size'])
    
#     print(transac_cost)
    df['Current Position'] = np.nan
    
    # Strongest trend
    df['Strongest Trend'] = (df.loc[:,['AD Trend', 'BP Trend', 'CD Trend', 'EC Trend', 'JY Trend', 'SF Trend']].abs().idxmax(axis=1)).str[:2]
    
    # Set initial position
    df.loc[start, 'Current Position'] = df.loc[start, 'Strongest Trend']
    df.loc[start, 'Take New Position'] = True
    
    # Pullback 
    conditions = [df['Strongest Trend'] == 'JY',
                  df['Strongest Trend'] == 'AD',
                  df['Strongest Trend'] == 'BP', 
                  df['Strongest Trend'] == 'CD',
                  df['Strongest Trend'] == 'EC',
                  df['Strongest Trend'] == 'SF']
#     values = [(df['JY Pullback Sign']),
#               df['AD Pullback Sign'],
#               df['BP Pullback Sign'],
#               df['CD Pullback Sign'],
#               df['EC Pullback Sign'],
#               (df['SF Pullback Sign'])]
    values = [(df['JY Pullback Sign'] & df['JY Pullback Thld']),
              df['AD Pullback Sign'] & df['AD Pullback Thld'],
              df['BP Pullback Sign'] & df['BP Pullback Thld'],
              df['CD Pullback Sign'] & df['CD Pullback Thld'],
              df['EC Pullback Sign'] & df['EC Pullback Thld'],
              (df['SF Pullback Sign'] & df['SF Pullback Thld'])]
                  
#     print(values[-1])
    df['Pullback'] = np.select(conditions, values, default="Invalid!")
#     df['Take New Position'] = np.where((df['Strongest Trend'] != df['Current Position']) & (df['Pullback']), True, False)
    df['Take New Position'] = np.where(df['Pullback'] == str(True), True, False)
#     print(np.any(df['Take New Position']))

    # Fill New Positions
    df['Current Position'] = np.where(df['Take New Position'], df['Strongest Trend'], df['Current Position'])
    
    # Fill holding positions 
    temp = df['Take New Position'].eq(1).cumsum()
#     print(df['Take New Position'].eq(1))
#     print(temp)
    df['Current Position'] = df.groupby(temp)['Current Position'].ffill()
#     print(df.loc[:, ['Current Position', 'Take New Position']].groupby(temp).ffill().head(30))
    
    df['Take New Position'] = np.where((df['Pullback'] == str(True)) & (df['Strongest Trend'] != (df['Current Position'].shift(1))), True, False)
    df.loc[start, 'Take New Position'] = True
                                       
    # Generate Position Size Held
    df['Position Size Held'] = np.nan
    pos_con = [df['Current Position'] == 'JY',
                  df['Current Position'] == 'AD',
                  df['Current Position'] == 'BP', 
                  df['Current Position'] == 'CD',
                  df['Current Position'] == 'EC',
                  df['Current Position'] == 'SF']
    pos_val = [(df['JY Theo Position Size']),
              df['AD Theo Position Size'],
              df['BP Theo Position Size'],
              df['CD Theo Position Size'],
              df['EC Theo Position Size'],
              (df['SF Theo Position Size'])]
    df['Position Size Held'] = np.where(df['Take New Position'] == True, np.select(pos_con, pos_val, default=np.nan), np.nan)
    
    temp2 = df['Current Position'].ne(df['Current Position'].shift(1)).cumsum()
#     print(temp2.iloc[15:35])
    df['Position Size Held'] = df.groupby(temp2)['Position Size Held'].ffill()
#     print(df.groupby(temp2)['Position Size Held'].head())
#     print(df['Position Size Held'])
    
    # Account for Transaction Costs (Multiplied by 2 since we have to both buy and sell the position)
    transac_con = [df['Take New Position'] & (df['Strongest Trend'] == 'JY'),
                  df['Take New Position'] & (df['Strongest Trend'] == 'AD'),
                  df['Take New Position'] & (df['Strongest Trend'] == 'BP'), 
                  df['Take New Position'] & (df['Strongest Trend'] == 'CD'),
                  df['Take New Position'] & (df['Strongest Trend'] == 'EC'),
                  df['Take New Position'] & (df['Strongest Trend'] == 'SF')]
    transac_values = [-2 * transac_cost['JY'] * df['JY Theo Position Size'],
                      -2 * transac_cost['AD'] * df['AD Theo Position Size'],
                      -2 * transac_cost['BP'] * df['BP Theo Position Size'],
                      -2 * transac_cost['CD'] * df['CD Theo Position Size'],
                      -2 * transac_cost['EC'] * df['EC Theo Position Size'],
                      -2 * transac_cost['SF'] * df['SF Theo Position Size']]
    df['Transac Cost'] = np.select(transac_con, transac_values, default=0)

    # Calculate P&L
    
    df['Total P&L'] = 0
    for market in market_list:
        mult_i = specs.loc[market, "Multiplier"]
        df[market+" Position One-Hot"] = np.where(df['Current Position'] == market, 1, 0)
        tmrw_bclose = df[market+"BClose"].shift(-1) 
        tdy_bclose = df[market+"BClose"]
        df[market+" P&L"] = (tmrw_bclose - tdy_bclose) * mult_i * df[market+" Position One-Hot"] * df['Position Size Held']
        df[market+" P&L"] = df[market+" P&L"].fillna(0)
        df[market+" P&L"] += df['Transac Cost'] * df[market+" Position One-Hot"]
#         print("P AND L")
#         print(df[market+" P&L"].cumsum())
        df['Total P&L'] += df[market+" P&L"].cumsum()
    
    return_df = df.copy()
    # Trim extraneous columns 
    for market in market_list:
        return_df.drop(columns=[market+'BClose',market+'SClose',
                                market+' Position One-Hot'], inplace=True)
    return return_df

post_trades = trade_logic(total)

print("Total Profit&Loss: " + str(post_trades.loc[post_trades.index[-1], "Total P&L"]))


post_trades.to_csv('output.csv', encoding='utf-8')
    
    