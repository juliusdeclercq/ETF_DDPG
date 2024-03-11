# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:20:23 2024

@author: Julius de Clercq
"""

import pandas as pd
import pandas_datareader.data as web
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Stooq data: https://stooq.com/db/h/
# NYSE ETFs: https://stooq.com/db/l/?g=70

symbols = ['ARKQ.US',   # ARK AUTONOMOUS TECHNOLOGY & ROBOTICS ETF
           'BATT.US',   # AMPLIFY LITHIUM & BATTERY TECHNOLOGY ETF
           'TAN.US',    # INVESCO SOLAR ETF
           'CNXT.US',   # VANECK CHINEXT ETF
           'CORN.US',   # TEUCRIUM CORN FUND
           'FUTY.US',   #  FIDELITY MSCI UTILITIES ETF
           'CRUZ.US',   # DEFIANCE HOTEL AIRLINE AND CRUISE ETF
           'DBEM.US',   # XTRACKERS MSCI EMERGING MARKETS HEDGED EQUITY ETF
           'COPX.US',   # GLOBAL X COPPER MINERS ETF
           'FRDM.US',   # FREEDOM 100 EMERGING MARKETS ETF
           'GCLN.US',   # GOLDMAN SACHS BLOOMBERG CLEAN ENERGY EQUITY ETF
           'GAMR.US',   # AMPLIFY VIDEO GAME TECH ETF
           'GBF.US',    # ISHARES GOVERNMENT/CREDIT BOND ETF
           'GCOW.US',   # PACER GLOBAL CASH COWS DIVIDEND ETF
           'INCO.US',   # COLUMBIA INDIA CONSUMER ETF
           'TEMP.US',   # JPMORGAN CLIMATE CHANGE SOLUTIONS ETF
           'THCX.US',   # AXS CANNABIS ETF
           'TFI.US',    # SPDR NUVEEN BLOOMBERG MUNICIPAL BOND ETF
           'TPHD.US',   # TIMOTHY PLAN HIGH DIVIDEND STOCK ETF
           'UGA.US',    # UNITED STATES GASOLINE FUND
           'UNG.US',    # UNITED STATES NATURAL GAS FUND
           'URTH.US',   # ISHARES MSCI WORLD ETF
           'VEGN.US',   # US VEGAN CLIMATE ETF
           'WUGI.US',   # AXS ESOTERICA NEXTG ECONOMY ETF
           'XSD.US',    # SPDR S&P SEMICONDUCTOR ETF
           'PSI.US',    # INVESCO SEMICONDUCTORS ETF
           'XLP.US',    # CONSUMER STAPLES SELECT SECTOR SPDR FUND
           'CRPT.US',   # FIRST TRUST SKYBRIDGE CRYPTO INDUSTRY AND DIGITAL ECONOMY ETF
           'PPTY.US',   # US DIVERSIFIED REAL ESTATE ETF
           'IHE.US',    # ISHARES US PHARMACEUTICALS ETF
           'DAT.US'    # PROSHARES BIG DATA REFINERS ETF
           ]
# print(len(symbols))
# etf_tickers   = list(pd.read_table("ETF_tickers.txt", sep = "\s\s+")["<TICKER>"])

df = web.DataReader(symbols, 'stooq', '2004-01-01', '2024-01-01')
df.to_csv("ETF_data_RAW.csv")



world_ticks_info = pd.read_table("World_index_tickers.txt", sep = "\s\s+")
world_tickers = list(world_ticks_info["<TICKER>"])
wdf = web.DataReader(world_tickers, 'stooq', '2004-01-01', '2024-01-01')


wdf = wdf.loc[:, wdf.columns.get_level_values('Attributes') == 'Close']
wdf = wdf.drop(columns = ['^CDAX', '^MT30', '^NOMUC'], level = 'Symbols', axis=1)
wdf.columns = wdf.columns.droplevel(level='Attributes')

# Remove the '^' prefix from the symbol names
wdf.columns = wdf.columns.str.replace('^', '')

wdf.to_csv("World_index_data.csv")















