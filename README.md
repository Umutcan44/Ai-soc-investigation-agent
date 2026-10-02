# AI SOC Investigation Agent

An AI-assisted Security Operations Center investigation pipeline for analyzing network security alerts using structured evidence, deterministic enrichment, and automated investigation workflows.

This project extends the network anomaly detection work from my Bachelor's Thesis by transforming machine-learning detection events into structured SOC investigation reports.

## Project Goal

The system is designed to process security events originating from:

- CICIDS2017
- CSE-CIC-IDS2018
- Network anomaly detection models
- Future SIEM/XDR integrations

The project separates deterministic security analysis from AI-generated investigation summaries.

## Architecture

```text
CICIDS2017 ──────────┐
CSE-CIC-IDS2018 ─────┤
Network Detector ─────┤
                     ↓
              SecurityEvent
                     ↓
               IOC Extraction
                     ↓
             Context Enrichment
                     ↓
            Investigation Engine
                     ↓
              MITRE ATT&CK
                     ↓
              AI SOC Agent
                     ↓
          Investigation Report
