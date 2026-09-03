import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from train import clean_text

# Set Page Config
st.set_page_config(
    page_title="SpamGuard Enterprise AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════
# PREMIUM 3D CINEMATIC UI — CSS DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* ─────────── GLOBAL DARK COSMOS THEME ─────────── */
    .stApp {
        background: #030712 !important;
        color: #E2E8F0 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse 80% 60% at 10% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 60%),
            radial-gradient(ellipse 60% 50% at 85% 70%, rgba(236, 72, 153, 0.06) 0%, transparent 55%),
            radial-gradient(ellipse 50% 40% at 50% 50%, rgba(6, 182, 212, 0.04) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }

    /* ─────────── ANIMATED BACKGROUND ORBS ─────────── */
    .ambient-orbs {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        overflow: hidden;
        pointer-events: none;
        z-index: 0;
    }
    .orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(80px);
        opacity: 0.35;
        animation: orbFloat 20s ease-in-out infinite alternate;
    }
    .orb-1 {
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.5) 0%, transparent 70%);
        top: -100px; left: -80px;
        animation-duration: 22s;
    }
    .orb-2 {
        width: 350px; height: 350px;
        background: radial-gradient(circle, rgba(236, 72, 153, 0.4) 0%, transparent 70%);
        bottom: -60px; right: -50px;
        animation-duration: 18s;
        animation-delay: -5s;
    }
    .orb-3 {
        width: 250px; height: 250px;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.35) 0%, transparent 70%);
        top: 40%; left: 50%;
        animation-duration: 25s;
        animation-delay: -10s;
    }
    @keyframes orbFloat {
        0%   { transform: translate(0, 0) scale(1); }
        33%  { transform: translate(40px, -30px) scale(1.1); }
        66%  { transform: translate(-20px, 25px) scale(0.95); }
        100% { transform: translate(15px, -15px) scale(1.05); }
    }

    /* ─────────── 3D FLOATING HEADER ─────────── */
    .header-3d {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.8rem;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        backdrop-filter: blur(40px) saturate(1.8);
        -webkit-backdrop-filter: blur(40px) saturate(1.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        margin-bottom: 1.8rem;
        box-shadow:
            0 4px 6px rgba(0, 0, 0, 0.3),
            0 10px 20px rgba(0, 0, 0, 0.25),
            0 25px 50px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
        animation: headerFloat 6s ease-in-out infinite;
        transform: perspective(1000px) rotateX(1deg);
        transform-style: preserve-3d;
    }
    .header-3d::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg,
            transparent 0%,
            rgba(99, 102, 241, 0.6) 20%,
            rgba(236, 72, 153, 0.6) 50%,
            rgba(6, 182, 212, 0.6) 80%,
            transparent 100%);
        border-radius: 20px 20px 0 0;
        animation: shimmerLine 4s linear infinite;
    }
    @keyframes headerFloat {
        0%, 100% { transform: perspective(1000px) rotateX(1deg) translateY(0); }
        50% { transform: perspective(1000px) rotateX(0.5deg) translateY(-4px); }
    }
    @keyframes shimmerLine {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }

    .brand-cluster {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-logo-3d {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #6366F1 0%, #EC4899 50%, #06B6D4 100%);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 1.4rem;
        color: #FFFFFF;
        box-shadow:
            0 0 20px rgba(99, 102, 241, 0.5),
            0 0 40px rgba(99, 102, 241, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        animation: logoPulse 3s ease-in-out infinite;
        transform: perspective(500px) rotateY(-5deg);
        transform-style: preserve-3d;
    }
    @keyframes logoPulse {
        0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.5), 0 0 40px rgba(99, 102, 241, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.3); }
        50% { box-shadow: 0 0 30px rgba(99, 102, 241, 0.7), 0 0 60px rgba(236, 72, 153, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.4); }
    }

    .brand-name-3d {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #FFFFFF 0%, #C7D2FE 40%, #A5B4FC 70%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: none;
        margin: 0;
    }
    .brand-sub {
        font-size: 0.7rem;
        font-weight: 500;
        color: #64748B;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin: 0;
    }

    .status-badge-3d {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #34D399;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 6px 14px;
        border-radius: 9999px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.1), inset 0 1px 0 rgba(16, 185, 129, 0.1);
    }
    .pulse-ring {
        position: relative;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981;
    }
    .pulse-ring::before {
        content: '';
        position: absolute;
        top: -3px; left: -3px;
        width: 14px; height: 14px;
        border-radius: 50%;
        border: 1.5px solid rgba(16, 185, 129, 0.5);
        animation: pulseRing 2s ease-out infinite;
    }
    @keyframes pulseRing {
        0% { transform: scale(0.8); opacity: 1; }
        100% { transform: scale(2); opacity: 0; }
    }

    /* ─────────── 3D METRIC CARDS ─────────── */
    .metric-grid-3d {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 2rem;
        perspective: 1200px;
    }
    @media (max-width: 900px) {
        .metric-grid-3d { grid-template-columns: repeat(2, 1fr); }
    }
    .metric-card-3d {
        position: relative;
        background: linear-gradient(165deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.6) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
        overflow: hidden;
        transition: transform 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.4s ease, border-color 0.4s ease;
        transform: perspective(800px) rotateX(0deg) rotateY(0deg) translateZ(0);
        transform-style: preserve-3d;
        box-shadow:
            0 4px 8px rgba(0, 0, 0, 0.3),
            0 8px 16px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.06);
        animation: cardEntrance 0.6s ease-out both;
    }
    .metric-card-3d:nth-child(1) { animation-delay: 0.1s; }
    .metric-card-3d:nth-child(2) { animation-delay: 0.2s; }
    .metric-card-3d:nth-child(3) { animation-delay: 0.3s; }
    .metric-card-3d:nth-child(4) { animation-delay: 0.4s; }

    @keyframes cardEntrance {
        from { opacity: 0; transform: perspective(800px) rotateX(10deg) translateY(30px) translateZ(-50px); }
        to { opacity: 1; transform: perspective(800px) rotateX(0) translateY(0) translateZ(0); }
    }

    .metric-card-3d:hover {
        transform: perspective(800px) rotateX(-3deg) rotateY(2deg) translateZ(15px) translateY(-6px);
        border-color: rgba(99, 102, 241, 0.35);
        box-shadow:
            0 8px 16px rgba(0, 0, 0, 0.3),
            0 16px 32px rgba(0, 0, 0, 0.2),
            0 0 40px rgba(99, 102, 241, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    /* Holographic top border per card */
    .metric-card-3d::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2.5px;
        border-radius: 18px 18px 0 0;
    }
    .metric-card-3d.accent-blue::before { background: linear-gradient(90deg, #38BDF8, #818CF8, #38BDF8); background-size: 200%; animation: holoSlide 3s linear infinite; }
    .metric-card-3d.accent-green::before { background: linear-gradient(90deg, #34D399, #06B6D4, #34D399); background-size: 200%; animation: holoSlide 3s linear infinite; animation-delay: -0.5s; }
    .metric-card-3d.accent-purple::before { background: linear-gradient(90deg, #A78BFA, #EC4899, #A78BFA); background-size: 200%; animation: holoSlide 3s linear infinite; animation-delay: -1s; }
    .metric-card-3d.accent-pink::before { background: linear-gradient(90deg, #F472B6, #FB923C, #F472B6); background-size: 200%; animation: holoSlide 3s linear infinite; animation-delay: -1.5s; }

    @keyframes holoSlide {
        0% { background-position: 0% 0; }
        100% { background-position: 200% 0; }
    }

    /* Inner glow orb per card */
    .metric-card-3d::after {
        content: '';
        position: absolute;
        width: 100px; height: 100px;
        border-radius: 50%;
        filter: blur(40px);
        opacity: 0.15;
        bottom: -20px; right: -20px;
        transition: opacity 0.4s;
    }
    .metric-card-3d.accent-blue::after { background: #38BDF8; }
    .metric-card-3d.accent-green::after { background: #34D399; }
    .metric-card-3d.accent-purple::after { background: #A78BFA; }
    .metric-card-3d.accent-pink::after { background: #F472B6; }
    .metric-card-3d:hover::after { opacity: 0.3; }

    .metric-icon-3d {
        width: 36px; height: 36px;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.15);
        transform: perspective(200px) rotateY(-5deg);
    }
    .icon-blue  { background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(129, 140, 248, 0.2)); border: 1px solid rgba(56, 189, 248, 0.25); }
    .icon-green { background: linear-gradient(135deg, rgba(52, 211, 153, 0.2), rgba(6, 182, 212, 0.2)); border: 1px solid rgba(52, 211, 153, 0.25); }
    .icon-purple { background: linear-gradient(135deg, rgba(167, 139, 250, 0.2), rgba(236, 72, 153, 0.2)); border: 1px solid rgba(167, 139, 250, 0.25); }
    .icon-pink  { background: linear-gradient(135deg, rgba(244, 114, 182, 0.2), rgba(251, 146, 60, 0.2)); border: 1px solid rgba(244, 114, 182, 0.25); }

    .metric-label-3d {
        font-size: 0.72rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
    }
    .metric-value-3d {
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .metric-sub-3d {
        font-size: 0.72rem;
        color: #64748B;
        margin-top: 6px;
        font-weight: 500;
    }

    /* ─────────── 3D TAB NAVIGATION ─────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.7) 0%, rgba(15, 23, 42, 0.5) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 6px 8px;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        perspective: 600px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        border-radius: 12px !important;
        padding: 9px 20px !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        transform: perspective(600px) translateZ(0);
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #CBD5E1 !important;
        background: rgba(255, 255, 255, 0.03) !important;
        transform: perspective(600px) translateZ(5px) !important;
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(236, 72, 153, 0.15) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.15), 0 0 20px rgba(99, 102, 241, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transform: perspective(600px) translateZ(8px) !important;
    }

    /* ─────────── 3D GLASS PANELS ─────────── */
    .glass-panel-3d {
        position: relative;
        background: linear-gradient(165deg, rgba(15, 23, 42, 0.75) 0%, rgba(30, 41, 59, 0.5) 100%);
        backdrop-filter: blur(30px) saturate(1.4);
        -webkit-backdrop-filter: blur(30px) saturate(1.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.4rem;
        box-shadow:
            0 4px 8px rgba(0, 0, 0, 0.25),
            0 12px 24px rgba(0, 0, 0, 0.15),
            0 24px 48px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.06);
        transition: transform 0.35s ease, box-shadow 0.35s ease;
        transform: perspective(1000px) rotateX(0deg);
    }
    .glass-panel-3d:hover {
        transform: perspective(1000px) rotateX(-1deg) translateY(-3px);
        box-shadow:
            0 6px 12px rgba(0, 0, 0, 0.3),
            0 16px 32px rgba(0, 0, 0, 0.2),
            0 32px 64px rgba(0, 0, 0, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }
    .glass-panel-3d::before {
        content: '';
        position: absolute;
        top: 0; left: 20px; right: 20px;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.12), transparent);
        border-radius: 20px 20px 0 0;
    }

    /* ─────────── 3D VERDICT BANNERS ─────────── */
    .verdict-3d-spam {
        position: relative;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(136, 19, 55, 0.2) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-left: 4px solid #EF4444;
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        margin: 1.2rem 0;
        box-shadow:
            0 4px 12px rgba(239, 68, 68, 0.08),
            0 12px 24px rgba(0, 0, 0, 0.15),
            0 0 60px rgba(239, 68, 68, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        animation: verdictSlide 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        transform: perspective(800px) rotateX(0deg);
        transform-style: preserve-3d;
    }
    .verdict-3d-ham {
        position: relative;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.06) 0%, rgba(6, 78, 59, 0.15) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 4px solid #10B981;
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        margin: 1.2rem 0;
        box-shadow:
            0 4px 12px rgba(16, 185, 129, 0.08),
            0 12px 24px rgba(0, 0, 0, 0.15),
            0 0 60px rgba(16, 185, 129, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        animation: verdictSlide 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        transform: perspective(800px) rotateX(0deg);
        transform-style: preserve-3d;
    }
    @keyframes verdictSlide {
        from { opacity: 0; transform: perspective(800px) rotateX(8deg) translateY(20px); }
        to { opacity: 1; transform: perspective(800px) rotateX(0deg) translateY(0); }
    }
    .verdict-title-3d {
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0 0 8px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .verdict-title-3d.spam { color: #FCA5A5; }
    .verdict-title-3d.ham { color: #6EE7B7; }
    .verdict-desc {
        font-size: 0.9rem;
        color: #CBD5E1;
        margin: 0;
        line-height: 1.6;
    }
    .verdict-stat-bar {
        margin-top: 14px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Floating 3D icon for verdict */
    .verdict-icon-float {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px; height: 38px;
        border-radius: 12px;
        font-size: 1.3rem;
        animation: iconBounce 2s ease-in-out infinite;
        transform: perspective(200px) rotateY(-5deg);
    }
    .verdict-icon-float.spam-icon {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
        border: 1px solid rgba(239, 68, 68, 0.3);
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.15);
    }
    .verdict-icon-float.ham-icon {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.1));
        border: 1px solid rgba(16, 185, 129, 0.3);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
    }
    @keyframes iconBounce {
        0%, 100% { transform: perspective(200px) rotateY(-5deg) translateY(0); }
        50% { transform: perspective(200px) rotateY(-5deg) translateY(-4px); }
    }

    /* ─────────── 3D TOKEN CHIPS ─────────── */
    .token-bar-3d {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin-top: 12px;
        perspective: 400px;
    }
    .token-3d-spam {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(220, 38, 38, 0.08));
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #FCA5A5;
        padding: 6px 14px;
        border-radius: 10px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        animation: chipPop 0.4s ease-out both;
    }
    .token-3d-ham {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.06));
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6EE7B7;
        padding: 6px 14px;
        border-radius: 10px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        animation: chipPop 0.4s ease-out both;
    }
    .token-3d-spam:hover, .token-3d-ham:hover {
        transform: perspective(400px) translateZ(8px) scale(1.05);
    }
    @keyframes chipPop {
        from { opacity: 0; transform: scale(0.8) translateY(8px); }
        to { opacity: 1; transform: scale(1) translateY(0); }
    }

    /* ─────────── 3D BUTTONS ─────────── */
    .stButton button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        transition: all 0.25s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25), 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        transform: perspective(500px) translateZ(0);
        position: relative;
    }
    .stButton button:hover {
        transform: perspective(500px) translateZ(6px) translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3), 0 8px 24px rgba(0, 0, 0, 0.2), 0 0 20px rgba(99, 102, 241, 0.1) !important;
    }
    .stButton button:active {
        transform: perspective(500px) translateZ(-2px) translateY(1px) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
    }

    /* ─────────── FORM CONTROLS 3D ─────────── */
    .stTextArea textarea {
        background: linear-gradient(165deg, rgba(15, 23, 42, 0.9) 0%, rgba(15, 23, 42, 0.7) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        color: #E2E8F0 !important;
        font-size: 0.95rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.3), 0 1px 0 rgba(255, 255, 255, 0.03) !important;
        transition: all 0.3s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.3), 0 0 0 2px rgba(99, 102, 241, 0.15), 0 0 30px rgba(99, 102, 241, 0.08) !important;
    }

    .stSlider [data-baseweb="slider"] {
        margin-top: 0.5rem;
    }

    /* ─────────── SECTION TITLES ─────────── */
    .section-title-3d {
        font-size: 0.8rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-title-3d .title-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-indigo { background: #818CF8; box-shadow: 0 0 8px rgba(129, 140, 248, 0.5); }
    .dot-cyan { background: #22D3EE; box-shadow: 0 0 8px rgba(34, 211, 238, 0.5); }
    .dot-rose { background: #FB7185; box-shadow: 0 0 8px rgba(251, 113, 133, 0.5); }
    .dot-amber { background: #FBBF24; box-shadow: 0 0 8px rgba(251, 191, 36, 0.5); }

    /* ─────────── 3D EXPANDER ─────────── */
    .stExpander {
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
        background: rgba(15, 23, 42, 0.5) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        margin-bottom: 8px !important;
        transition: transform 0.2s ease !important;
    }
    .stExpander:hover {
        transform: translateY(-1px) !important;
        border-color: rgba(99, 102, 241, 0.2) !important;
    }

    /* ─────────── PROGRESS BAR ─────────── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366F1, #EC4899, #EF4444) !important;
        border-radius: 8px !important;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.3) !important;
    }

    /* ─────────── FILE UPLOADER ─────────── */
    .stFileUploader {
        border-radius: 16px !important;
    }

    /* ─────────── DECORATIVE 3D SHAPES ─────────── */
    .deco-cube {
        position: absolute;
        width: 20px; height: 20px;
        border: 1.5px solid rgba(99, 102, 241, 0.2);
        border-radius: 4px;
        transform: rotate(45deg);
        animation: cubeFloat 8s ease-in-out infinite;
    }
    @keyframes cubeFloat {
        0%, 100% { transform: rotate(45deg) translateY(0) scale(1); opacity: 0.3; }
        50% { transform: rotate(225deg) translateY(-10px) scale(1.1); opacity: 0.5; }
    }

    .deco-hexagon {
        position: absolute;
        width: 16px; height: 16px;
        background: rgba(236, 72, 153, 0.08);
        clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
        animation: hexSpin 12s linear infinite;
    }
    @keyframes hexSpin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* ─────────── GLOBAL ANIMATIONS ─────────── */
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes glowPulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }

    /* Scrollbar Styling */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.5); }
    ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.5); }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model_path = os.path.join("models", "spam_model.pkl")
    vec_path = os.path.join("models", "tfidf_vectorizer.pkl")
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return None, None
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


@st.cache_data
def load_and_prepare_dataset():
    data_path = os.path.join("data", "spam.csv")
    if not os.path.exists(data_path):
        return None, None, None, None, None
    df = pd.read_csv(data_path)
    df["cleaned"] = df["message"].apply(clean_text)
    df["char_len"] = df["message"].apply(len)
    df["word_len"] = df["message"].apply(lambda x: len(str(x).split()))
    
    X = df["cleaned"]
    y = df["label"].map({"ham": 0, "spam": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return df, X_train, X_test, y_train, y_test


def analyze_message_tokens(cleaned_text, vectorizer, model):
    words = cleaned_text.split()
    if not words or not hasattr(vectorizer, 'vocabulary_'):
        return []

    vocab = vectorizer.vocabulary_
    ham_log_probs = model.feature_log_prob_[0]
    spam_log_probs = model.feature_log_prob_[1]

    found_triggers = []
    seen = set()
    for word in words:
        if word in vocab and word not in seen:
            seen.add(word)
            idx = vocab[word]
            spam_lp = spam_log_probs[idx]
            ham_lp = ham_log_probs[idx]
            log_odds = spam_lp - ham_lp
            found_triggers.append({
                "word": word,
                "spam_score": spam_lp,
                "log_odds": log_odds,
                "leaning": "Spam" if log_odds > 0 else "Ham"
            })

    found_triggers.sort(key=lambda x: abs(x["log_odds"]), reverse=True)
    return found_triggers[:8]


def set_input_text(text: str):
    st.session_state["user_message_input"] = text


def clear_input_text():
    st.session_state["user_message_input"] = ""


def main():
    model, vectorizer = load_artifacts()
    df, X_train, X_test, y_train, y_test = load_and_prepare_dataset()

    if "user_message_input" not in st.session_state:
        st.session_state["user_message_input"] = ""

    # ═══ Ambient 3D Background Orbs ═══
    st.markdown("""
    <div class="ambient-orbs">
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
        <div class="orb orb-3"></div>
    </div>
    """, unsafe_allow_html=True)

    # ═══ 3D Floating Header ═══
    st.markdown("""
    <div class="header-3d">
        <div class="brand-cluster">
            <div class="brand-logo-3d">⚡</div>
            <div>
                <h1 class="brand-name-3d">SpamGuard Pro</h1>
                <p class="brand-sub">Enterprise AI Detection Engine</p>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:16px;">
            <div class="status-badge-3d">
                <span class="pulse-ring"></span>
                <span>MODEL ONLINE (v2.4)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ═══ 3D Metric Cards with Icons ═══
    st.markdown("""
    <div class="metric-grid-3d">
        <div class="metric-card-3d accent-blue">
            <div class="metric-icon-3d icon-blue">🎯</div>
            <div class="metric-label-3d">Overall Accuracy</div>
            <div class="metric-value-3d" style="color:#38BDF8;">98.21%</div>
            <div class="metric-sub-3d">Stratified 20% Test Split</div>
        </div>
        <div class="metric-card-3d accent-green">
            <div class="metric-icon-3d icon-green">🛡️</div>
            <div class="metric-label-3d">Spam Precision</div>
            <div class="metric-value-3d" style="color:#34D399;">97.78%</div>
            <div class="metric-sub-3d">Ultra-low false alarms</div>
        </div>
        <div class="metric-card-3d accent-purple">
            <div class="metric-icon-3d icon-purple">📡</div>
            <div class="metric-label-3d">Spam Recall</div>
            <div class="metric-value-3d" style="color:#A78BFA;">88.59%</div>
            <div class="metric-sub-3d">High catch rate</div>
        </div>
        <div class="metric-card-3d accent-pink">
            <div class="metric-icon-3d icon-pink">⚡</div>
            <div class="metric-label-3d">Latency</div>
            <div class="metric-value-3d" style="color:#F472B6;">&lt; 3.2ms</div>
            <div class="metric-sub-3d">Real-time inference</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Modern Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚡ Real-Time Scanner",
        "📂 Batch File Analyzer",
        "📊 Dynamic Pipeline & Confusion Matrix",
        "💡 Defense & Architecture Guide"
    ])

    # ==========================================
    # TAB 1: REAL-TIME SCANNER
    # ==========================================
    with tab1:
        if model is None or vectorizer is None:
            st.error("⚠️ Model artifacts missing. Please train model using `python src/train.py`.")
            return

        col_left, col_right = st.columns([1.7, 1], gap="large")

        with col_right:
            st.markdown("""
            <div class="glass-panel-3d" style="padding:1.3rem;">
                <div class="section-title-3d">
                    <span class="title-dot dot-cyan"></span>
                    Quick Test Vectors
                </div>
            """, unsafe_allow_html=True)

            p1 = "URGENT! You have won £1,000 cash prize! Claim your reward now by texting CLAIM to 87121."
            p2 = "Hey, are you free for lunch tomorrow at 1 PM near the library?"
            p3 = "Congratulations! You have been selected for a free $500 Amazon gift card. Click http://bit.ly/gift to verify."
            p4 = "Hi team, please find attached the weekly sales report and presentation slides."
            p5 = "FINAL NOTICE: Your credit account is blocked. Verify your details immediately to avoid fees."

            st.button("🚨 Urgent Cash Prize (Spam)", on_click=set_input_text, args=(p1,), width="stretch")
            st.button("👥 Lunch Invitation (Ham)", on_click=set_input_text, args=(p2,), width="stretch")
            st.button("🎁 $500 Gift Card Phishing (Spam)", on_click=set_input_text, args=(p3,), width="stretch")
            st.button("💼 Office Weekly Report (Ham)", on_click=set_input_text, args=(p4,), width="stretch")
            st.button("⚠️ Account Block Notice (Spam)", on_click=set_input_text, args=(p5,), width="stretch")

            st.markdown("</div>", unsafe_allow_html=True)

        with col_left:
            st.markdown("""
            <div class="section-title-3d">
                <span class="title-dot dot-indigo"></span>
                Input Text Message / Email
            </div>
            """, unsafe_allow_html=True)

            user_text = st.text_area(
                "Message Body:",
                height=140,
                placeholder="Type or paste any SMS, email, or suspicious message text here...",
                key="user_message_input",
                label_visibility="collapsed"
            )

            c_btn1, c_btn2, c_btn3 = st.columns([1.3, 1.3, 2.5])
            with c_btn1:
                scan_btn = st.button("🚀 Analyze Now", type="primary", width="stretch")
            with c_btn2:
                st.button("🧹 Clear Input", on_click=clear_input_text, width="stretch")

            if user_text.strip():
                cleaned = clean_text(user_text)
                vec_features = vectorizer.transform([cleaned])
                pred = model.predict(vec_features)[0]
                probabilities = model.predict_proba(vec_features)[0]

                ham_pct = probabilities[0] * 100
                spam_pct = probabilities[1] * 100

                if pred == 1:
                    st.markdown(f"""
                    <div class="verdict-3d-spam">
                        <div class="verdict-title-3d spam">
                            <span class="verdict-icon-float spam-icon">🚨</span>
                            <span>CRITICAL: SPAM / FRAUD DETECTED</span>
                        </div>
                        <p class="verdict-desc">High concentration of deceptive indicators, phishing tokens, or urgent marketing patterns found.</p>
                        <div class="verdict-stat-bar" style="color:#FCA5A5;">
                            Spam Probability: <b>{spam_pct:.2f}%</b> &nbsp;│&nbsp; Legitimate: {ham_pct:.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-3d-ham">
                        <div class="verdict-title-3d ham">
                            <span class="verdict-icon-float ham-icon">🛡️</span>
                            <span>AUTHENTIC / LEGITIMATE (HAM)</span>
                        </div>
                        <p class="verdict-desc">Message appears safe, standard, and appropriate for primary inbox delivery.</p>
                        <div class="verdict-stat-bar" style="color:#6EE7B7;">
                            Legitimate Probability: <b>{ham_pct:.2f}%</b> &nbsp;│&nbsp; Spam: {spam_pct:.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.progress(float(spam_pct / 100.0), text=f"Spam Likelihood: {spam_pct:.1f}%")

                triggers = analyze_message_tokens(cleaned, vectorizer, model)
                if triggers:
                    st.markdown("<div class='section-title-3d' style='margin-top:16px;'><span class='title-dot dot-rose'></span> Influential Word Tokens</div>", unsafe_allow_html=True)
                    chips_html = '<div class="token-bar-3d">' + "".join([
                        f'<span class="{"token-3d-spam" if t["leaning"] == "Spam" else "token-3d-ham"}" style="animation-delay:{i*0.06}s">'
                        f'{"🔴" if t["leaning"] == "Spam" else "🟢"} {t["word"]} ({t["leaning"]})'
                        f'</span>' for i, t in enumerate(triggers)
                    ]) + '</div>'
                    st.markdown(chips_html, unsafe_allow_html=True)

                    with st.expander("📊 View Token Log-Odds Weights"):
                        chart_df = pd.DataFrame(triggers)
                        fig, ax = plt.subplots(figsize=(7, 2.8), facecolor='#0B0F19')
                        ax.set_facecolor('#0B0F19')
                        colors = ['#EF4444' if x > 0 else '#10B981' for x in chart_df['log_odds']]
                        ax.barh(chart_df['word'], chart_df['log_odds'], color=colors, height=0.55)
                        ax.axvline(0, color='#475569', linestyle='--', linewidth=0.8)
                        ax.tick_params(colors='#94A3B8', labelsize=8)
                        ax.set_xlabel("Relative Weight (>0 Spam | <0 Ham)", color='#94A3B8', fontsize=8)
                        ax.invert_yaxis()
                        for spine in ax.spines.values():
                            spine.set_color('#1E293B')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
            elif scan_btn:
                st.warning("Please type or select a message to analyze.")

    # ==========================================
    # TAB 2: BATCH FILE PROCESSING
    # ==========================================
    with tab2:
        st.markdown("""
        <div class="glass-panel-3d">
            <div class="section-title-3d" style="margin-bottom:4px;"><span class="title-dot dot-amber"></span> Bulk File Processing</div>
            <h3 style="margin-top:0; font-size:1.2rem; font-weight:700; color:#FFFFFF;">📂 Upload & Classify at Scale</h3>
            <p style="color:#94A3B8; font-size:0.9rem; margin-bottom:1rem;">Upload large CSV or TXT datasets for parallelized high-throughput classification.</p>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload CSV / TXT", type=["csv", "txt"], label_visibility="collapsed")

        if uploaded_file is not None and model is not None and vectorizer is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    batch_df = pd.read_csv(uploaded_file)
                else:
                    lines = uploaded_file.read().decode("utf-8").splitlines()
                    batch_df = pd.DataFrame({"message": [l for l in lines if l.strip()]})

                if "message" not in batch_df.columns:
                    batch_df.rename(columns={batch_df.columns[0]: "message"}, inplace=True)

                cleaned_series = batch_df["message"].apply(clean_text)
                batch_vec = vectorizer.transform(cleaned_series)
                batch_preds = model.predict(batch_vec)
                batch_probs = model.predict_proba(batch_vec)

                batch_df["Verdict"] = ["SPAM 🔴" if p == 1 else "HAM 🟢" for p in batch_preds]
                batch_df["Spam Confidence (%)"] = [round(prob[1] * 100, 2) for prob in batch_probs]

                st.success(f"Processed {len(batch_df)} messages successfully!")
                
                b_c1, b_c2, b_c3 = st.columns(3)
                spam_count = int((batch_preds == 1).sum())
                ham_count = int((batch_preds == 0).sum())
                b_c1.metric("Total Processed", len(batch_df))
                b_c2.metric("Spam Identified", f"{spam_count} ({spam_count/len(batch_df)*100:.1f}%)")
                b_c3.metric("Ham Identified", f"{ham_count} ({ham_count/len(batch_df)*100:.1f}%)")

                st.dataframe(batch_df[["message", "Verdict", "Spam Confidence (%)"]], width="stretch")

                csv_data = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Classified CSV",
                    data=csv_data,
                    file_name="spamguard_batch_results.csv",
                    mime="text/csv",
                    type="primary"
                )
            except Exception as e:
                st.error(f"Error processing file: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # TAB 3: DYNAMIC PIPELINE & CONFUSION MATRIX
    # ==========================================
    with tab3:
        if model is not None and vectorizer is not None and X_test is not None:
            X_test_vec = vectorizer.transform(X_test)
            y_test_probs = model.predict_proba(X_test_vec)[:, 1]

            ctrl_col, plot_col = st.columns([1.1, 1.4], gap="large")

            with ctrl_col:
                st.markdown("""
                <div class="glass-panel-3d">
                    <div class="section-title-3d"><span class="title-dot dot-indigo"></span> Threshold Control</div>
                    <h4 style="margin-top:0; font-size:1.05rem; font-weight:700; color:#FFFFFF;">🎛️ Dynamic Decision Threshold</h4>
                    <p style="color:#94A3B8; font-size:0.85rem; margin-bottom:12px;">Adjust threshold to tune spam strictness and observe real-time matrix changes.</p>
                """, unsafe_allow_html=True)

                threshold = st.slider(
                    "Threshold",
                    min_value=0.01,
                    max_value=0.99,
                    value=0.50,
                    step=0.01,
                    label_visibility="collapsed"
                )

                y_pred_dynamic = (y_test_probs >= threshold).astype(int)
                cm_dyn = confusion_matrix(y_test, y_pred_dynamic)
                acc_dyn = accuracy_score(y_test, y_pred_dynamic)
                prec_dyn = precision_score(y_test, y_pred_dynamic, zero_division=0)
                rec_dyn = recall_score(y_test, y_pred_dynamic, zero_division=0)
                f1_dyn = f1_score(y_test, y_pred_dynamic, zero_division=0)

                tn, fp, fn, tp = cm_dyn.ravel() if cm_dyn.shape == (2, 2) else (0, 0, 0, 0)

                m1, m2 = st.columns(2)
                m1.metric("Live Accuracy", f"{acc_dyn * 100:.2f}%")
                m2.metric("Spam Precision", f"{prec_dyn * 100:.2f}%")
                m1.metric("Spam Recall", f"{rec_dyn * 100:.2f}%")
                m2.metric("F1-Score", f"{f1_dyn * 100:.2f}%")

                st.markdown(f"""
                <div style="background:linear-gradient(165deg, rgba(15,23,42,0.9) 0%, rgba(15,23,42,0.7) 100%); border:1px solid rgba(255,255,255,0.08); padding:14px; border-radius:14px; margin-top:14px; font-family:'JetBrains Mono', monospace; font-size:0.82rem; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2), 0 1px 0 rgba(255,255,255,0.03);">
                    <div style="color:#34D399; margin-bottom:6px; display:flex; align-items:center; gap:6px;"><b>🟢 True Ham (Inbox Allowed):</b> {tn}</div>
                    <div style="color:#F87171; margin-bottom:6px; display:flex; align-items:center; gap:6px;"><b>⚠️ False Alarm (False Positives):</b> {fp}</div>
                    <div style="color:#FBBF24; margin-bottom:6px; display:flex; align-items:center; gap:6px;"><b>⚠️ Missed Spam (False Negatives):</b> {fn}</div>
                    <div style="color:#60A5FA; display:flex; align-items:center; gap:6px;"><b>🔴 True Spam (Correctly Caught):</b> {tp}</div>
                </div>
                </div>
                """, unsafe_allow_html=True)

            with plot_col:
                st.markdown("""
                <div class="glass-panel-3d">
                    <div class="section-title-3d"><span class="title-dot dot-rose"></span> Live Visualization</div>
                    <h4 style="margin-top:0; font-size:1.05rem; font-weight:700; color:#FFFFFF;">🎯 Real-Time Confusion Matrix</h4>
                """, unsafe_allow_html=True)
                
                fig_cm, ax_cm = plt.subplots(figsize=(6, 3.8), facecolor='#0B0F19')
                ax_cm.set_facecolor('#0B0F19')
                sns.heatmap(
                    cm_dyn,
                    annot=True,
                    fmt="d",
                    cmap="Reds" if threshold > 0.5 else "Blues",
                    cbar=False,
                    xticklabels=["Predicted Ham", "Predicted Spam"],
                    yticklabels=["Actual Ham", "Actual Spam"],
                    annot_kws={"size": 13, "weight": "bold", "color": "#FFFFFF"},
                    ax=ax_cm
                )
                ax_cm.tick_params(colors='#94A3B8', labelsize=9)
                ax_cm.set_title(f"Confusion Matrix (Decision Threshold = {threshold:.2f})", fontsize=10, color='#E2E8F0', pad=10)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close()
                st.markdown("</div>", unsafe_allow_html=True)

            # Secondary Visuals
            v_c1, v_c2 = st.columns(2)
            with v_c1:
                st.markdown("""<div class="glass-panel-3d">
                <div class="section-title-3d"><span class="title-dot dot-cyan"></span> Message Length Distribution</div>
                """, unsafe_allow_html=True)
                fig_len, ax_len = plt.subplots(figsize=(6, 3.2), facecolor='#0B0F19')
                ax_len.set_facecolor('#0B0F19')
                sns.histplot(data=df, x="char_len", hue="label", bins=40, palette={"ham": "#10B981", "spam": "#EF4444"}, kde=True, ax=ax_len)
                ax_len.set_xlim(0, 300)
                ax_len.tick_params(colors='#94A3B8', labelsize=8)
                ax_len.set_xlabel("Characters", color='#94A3B8', fontsize=8)
                ax_len.set_ylabel("Count", color='#94A3B8', fontsize=8)
                for spine in ax_len.spines.values():
                    spine.set_color('#1E293B')
                plt.tight_layout()
                st.pyplot(fig_len)
                plt.close()
                st.markdown("</div>", unsafe_allow_html=True)

            with v_c2:
                st.markdown("""<div class="glass-panel-3d">
                <div class="section-title-3d"><span class="title-dot dot-rose"></span> Top Indicative Spam Terms</div>
                """, unsafe_allow_html=True)
                vocab = vectorizer.vocabulary_
                spam_log_probs = model.feature_log_prob_[1]
                inv_vocab = {v: k for k, v in vocab.items()}
                top_indices = np.argsort(spam_log_probs)[-10:]
                top_words = [inv_vocab[i] for i in top_indices]
                top_scores = [spam_log_probs[i] for i in top_indices]

                fig_top, ax_top = plt.subplots(figsize=(6, 3.2), facecolor='#0B0F19')
                ax_top.set_facecolor('#0B0F19')
                ax_top.barh(top_words, top_scores, color="#EF4444", height=0.55)
                ax_top.tick_params(colors='#94A3B8', labelsize=8)
                ax_top.set_xlabel("Spam Log Probability", color='#94A3B8', fontsize=8)
                for spine in ax_top.spines.values():
                    spine.set_color('#1E293B')
                plt.tight_layout()
                st.pyplot(fig_top)
                plt.close()
                st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # TAB 4: DEFENSE & ARCHITECTURE GUIDE
    # ==========================================
    with tab4:
        st.markdown("""
        <div class="glass-panel-3d">
            <div class="section-title-3d" style="margin-bottom:6px;"><span class="title-dot dot-amber"></span> Knowledge Base</div>
            <h3 style="margin-top:0; font-size:1.2rem; font-weight:700; color:#FFFFFF;">💡 Project Architecture & Viva Defense Reference</h3>
            <p style="color:#94A3B8; font-size:0.9rem; margin-bottom:1rem;">Essential machine learning principles and architectural justifications.</p>
        """, unsafe_allow_html=True)

        with st.expander("❓ 1. What is the fundamental difference between Ham and Spam?"):
            st.write(
                "**Ham** represents authentic, expected communications (e.g. personal messages, team updates, university notifications).\n\n"
                "**Spam** refers to unsolicited bulk messages containing financial fraud, phishing links, or deceptive advertising."
            )

        with st.expander("❓ 2. How does TF-IDF feature extraction work?"):
            st.write(
                "- **TF (Term Frequency):** Frequency of a token in a specific message.\n"
                "- **IDF (Inverse Document Frequency):** Reduces the importance of frequent stop-words and boosts discriminative keywords (*urgent*, *claim*, *reward*, *prize*).\n"
                "- Converts unstructured text into mathematical vectors."
            )

        with st.expander("❓ 3. Why Multinomial Naive Bayes instead of complex Deep Learning?"):
            st.write(
                "- **Sub-5ms Inference:** Operates in milliseconds with minimal CPU overhead.\n"
                "- **High Benchmark Score:** Achieved **98.21% Accuracy** and **97.78% Precision** without requiring heavy GPU clusters."
            )

        with st.expander("❓ 4. Why is Precision prioritized over Recall?"):
            st.write(
                "In spam filtering, a **False Positive** (wrongfully discarding a legitimate email) causes severe real-world disruption compared to a **False Negative** (a minor spam item slipping into the inbox)."
            )

        with st.expander("❓ 5. What role does Laplace Smoothing (alpha=0.1) play?"):
            st.write(
                "Prevents the **Zero Probability Problem** when encountering unseen vocabulary tokens in new incoming messages."
            )

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
