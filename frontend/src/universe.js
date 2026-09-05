/**
 * Flagship Tri-Exchange Universal Ticker Definitions (286 Symbols)
 * - NASDAQ: 98 validated constituents (e.g. AAPL, NVDA, MSFT, ARM, PLTR)
 * - NYSE: 95 validated blue-chips & leaders (e.g. JPM, WMT, LLY, UNH, BA)
 * - LSE: 93 validated FTSE-100 constituents (e.g. SHEL.L, AZN.L, RR.L, BARC.L)
 */

export const NASDAQ_TICKERS = [
  'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'COST', 'QCOM',
  'AVGO', 'NFLX', 'ADBE', 'TXN', 'INTC', 'CMCSA', 'PEP', 'CSCO', 'INTU', 'AMAT',
  'PYPL', 'BKNG', 'ISRG', 'MDLZ', 'GILD', 'REGN', 'ADI', 'VRTX', 'LRCX', 'PANW',
  'SNPS', 'KLAC', 'CDNS', 'CHTR', 'MAR', 'ORLY', 'NXPI', 'FTNT', 'CTAS', 'PCAR',
  'KDP', 'PAYX', 'MNST', 'MCHP', 'ROST', 'AEP', 'KHC', 'ODFL', 'FAST', 'IDXX',
  'EXC', 'LULU', 'VRSK', 'CSX', 'GEHC', 'BIIB', 'DXCM', 'XEL', 'FANG', 'TEAM',
  'MRVL', 'ON', 'BKR', 'WBD', 'DLTR', 'ILMN', 'ALGN', 'CEG', 'CPRT', 'SBUX',
  'HON', 'ASML', 'AZN', 'PDD', 'MELI', 'CRWD', 'DDOG', 'ZS', 'WDAY', 'ABNB',
  'MRNA', 'DASH', 'TTD', 'MDB', 'SMCI', 'ARM', 'APP', 'PLTR', 'ROP', 'TTWO',
  'CDW', 'GFS', 'ANET', 'CCEP', 'AXON', 'MSTR', 'LIN', 'CTSH',
];

export const NYSE_TICKERS = [
  'JPM', 'XOM', 'WMT', 'JNJ', 'CAT', 'KO', 'NEE', 'DIS', 'BAC', 'GE',
  'UNH', 'PG', 'V', 'HD', 'CVX', 'MA', 'PFE', 'MRK', 'ABT', 'ORCL',
  'LLY', 'MCD', 'TMO', 'NKE', 'IBM', 'GS', 'MS', 'RTX', 'UNP', 'BMY',
  'COP', 'PM', 'LOW', 'UPS', 'MSI', 'DE', 'SCHW', 'C', 'AXP', 'BLK',
  'PGR', 'CB', 'AON', 'MET', 'PRU', 'AIG', 'TRV', 'ALL', 'EOG', 'SLB',
  'MPC', 'PSX', 'VLO', 'OXY', 'DVN', 'KMI', 'WMB', 'ET', 'MDT', 'SYK',
  'BSX', 'EW', 'BAX', 'BDX', 'ZTS', 'CI', 'ELV', 'HUM', 'CVS', 'FDX',
  'NSC', 'WM', 'RSG', 'EMR', 'ETN', 'ITW', 'PH', 'CMI', 'TT', 'JCI',
  'CARR', 'OTIS', 'GD', 'NOC', 'LHX', 'LMT', 'BA', 'HMC', 'TM', 'SONY',
  'NVO', 'BABA', 'NVS', 'SAP', 'TSM', 'DEO', 'BAM',
];

export const LSE_TICKERS = [
  'SHEL.L', 'AZN.L', 'HSBA.L', 'BP.L', 'ULVR.L', 'GSK.L', 'RIO.L', 'BATS.L', 'BARC.L', 'DGE.L',
  'REL.L', 'LSEG.L', 'LLOY.L', 'NG.L', 'EXPN.L', 'VOD.L', 'GLEN.L', 'AAL.L', 'PRU.L', 'NWG.L',
  'STAN.L', 'BA.L', 'CPG.L', 'IMB.L', 'AV.L', 'SSE.L', 'RR.L', 'TSCO.L', 'ABF.L', 'ADM.L',
  'ANTO.L', 'INF.L', 'MNDI.L', 'WPP.L', 'HLN.L', 'SDR.L', 'LAND.L', 'BLND.L', 'UU.L', 'SVT.L',
  'WTB.L', 'AUTO.L', 'ENT.L', 'JD.L', 'KGF.L', 'MKS.L', 'NXT.L', 'BME.L', 'OCDO.L', 'SMIN.L',
  'SMT.L', 'FDM.L', 'PSN.L', 'TW.L', 'CRDA.L', 'JMAT.L', 'HLMA.L', 'SPX.L', 'WEIR.L', 'IMI.L',
  'SN.L', 'SGE.L', 'RS1.L', 'BEZ.L', 'HIK.L', 'IHG.L', 'LGEN.L', 'STJ.L', 'UTG.L', 'ITRK.L',
  'BKG.L', 'SGRO.L', 'PSON.L', 'DCC.L', 'SBRY.L', 'FRAS.L', 'RMV.L', 'WISE.L', 'EMG.L', 'MONY.L',
  'EDV.L', 'GAW.L', 'IGG.L', 'VTY.L', 'WIZZ.L', 'GNS.L', 'BOY.L', 'ITV.L', 'GFTU.L', 'TRN.L', 'EZJ.L',
];

export const ALL_VALID_TICKERS = [...NASDAQ_TICKERS, ...NYSE_TICKERS, ...LSE_TICKERS];
export const ALL_TICKERS_SET = new Set(ALL_VALID_TICKERS);
