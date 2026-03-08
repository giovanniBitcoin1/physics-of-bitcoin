#!/usr/bin/env python3
"""
generate_figure.py
Fetches BTC price data from Newhedge, fits the power law, saves PNG.
Called daily by GitHub Actions.
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import date
import os, sys

API_KEY  = os.environ.get('NEWHEDGE_API_KEY', '')
GENESIS  = date(2009, 1, 3)

# ── Fetch ──────────────────────────────────────────────────────────────────
def fetch_price():
    # Public endpoint — no token required for price data
    r = requests.get(
        'https://newhedge.io/api/v1/analytics/bitcoin_live_price',
        params={'interval': '1_day', 'ytdMode': 'off'},
        timeout=30)
    r.raise_for_status()
    # Response: {"data": [{"p": price, "t": timestamp_seconds}, ...]}
    data = r.json()['data']
    df = pd.DataFrame(data)
    df['date']  = pd.to_datetime(df['t'], unit='s').dt.date
    df['close'] = df['p'].astype(float)
    return df[['date', 'close']].dropna().sort_values('date').reset_index(drop=True)

# ── Figure ─────────────────────────────────────────────────────────────────
def make_figure(df, out='docs/bitcoin_powerlaw.png'):
    df = df.copy()
    df['t'] = df['date'].apply(lambda d: (d - GENESIS).days).astype(float)
    df = df[(df['t'] > 0) & (df['close'] > 0)].reset_index(drop=True)

    t     = df['t'].values
    price = df['close'].values

    # Power law fit in log-log space
    log_t = np.log10(t)
    log_p = np.log10(price)
    coeffs    = np.polyfit(log_t, log_p, 1)
    slope     = coeffs[0]
    intercept = coeffs[1]

    price_fit = 10 ** (slope * log_t + intercept)
    residuals = log_p - np.log10(price_fit)
    sigma     = np.std(residuals)

    band_hi = price_fit * 10 ** ( 2 * sigma)
    band_lo = price_fit * 10 ** (-2 * sigma)

    # ── Style ──────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Georgia', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'text.color': '#1a1a2e',
        'axes.labelcolor': '#1a1a2e',
        'xtick.color': '#1a1a2e',
        'ytick.color': '#1a1a2e',
    })

    DARK  = '#1a1a2e'
    BLUE  = '#2a5fa8'
    GREY  = '#7f8c8d'
    LGREY = '#dde3e8'
    RED   = '#c0392b'

    fig, ax = plt.subplots(figsize=(10, 6.5))
    plt.subplots_adjust(left=0.11, right=0.93, top=0.88, bottom=0.13)

    ax.loglog(t, price,     '.', color=GREY, markersize=1.8, alpha=0.5,
              zorder=3, label='Daily close (BTC/USD)')
    ax.loglog(t, price_fit, '-', color=BLUE, linewidth=2.2,
              zorder=5, label=f'Power law fit  β = {slope:.2f}')
    ax.fill_between(t, band_lo, band_hi,
                    color=BLUE, alpha=0.08, zorder=2,
                    label=r'±2σ band')
    ax.loglog(t[-1], price[-1], 'o', color=RED, markersize=8,
              markeredgecolor=DARK, markeredgewidth=0.8,
              zorder=8, label=f'Today  ${price[-1]:,.0f}')

    # X ticks as calendar years
    year_ticks = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
    t_ticks    = [(date(y,1,1) - GENESIS).days for y in year_ticks]
    ax.set_xticks(t_ticks)
    ax.set_xticklabels([str(y) for y in year_ticks], fontsize=10)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'${x:,.0f}' if x >= 1 else f'${x:.4f}'))

    ax.set_xlabel('Year', fontsize=12, labelpad=8)
    ax.set_ylabel('BTC / USD  (log scale)', fontsize=12, labelpad=8)
    ax.set_title(
        f'Bitcoin Price Power Law  —  P(t) ∝ t^{slope:.2f}\n'
        f'{len(df):,} daily observations · genesis to {df["date"].iloc[-1]}',
        fontsize=10.5, color=DARK, pad=10, loc='left', style='italic')

    ax.text(0.99, 0.03,
            f'Updated {date.today()}  ·  thephysicsofbitcoin.com',
            ha='right', va='bottom', fontsize=8, color=GREY,
            transform=ax.transAxes, style='italic')

    ax.legend(loc='upper left', fontsize=9.5, frameon=False,
              handlelength=1.8, bbox_to_anchor=(0.01, 0.99))

    for yv in [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]:
        ax.axhline(yv, color=LGREY, linewidth=0.4, zorder=1)

    os.makedirs('docs', exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved → {out}   slope={slope:.3f}   price=${price[-1]:,.0f}')

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Fetching data...')
    df = fetch_price()
    print(f'  {len(df)} rows  ({df["date"].iloc[0]} -> {df["date"].iloc[-1]})')
    make_figure(df)
    print('Done.')
