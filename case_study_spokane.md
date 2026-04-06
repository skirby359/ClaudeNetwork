# What 5.9 Million Emails Reveal About Spokane County — Without Reading a Single One

**An organizational communication analysis using only email metadata (Date, Size, From, To). No email bodies were accessed.**

---

## The Dataset

Spokane County, Washington. 5.9 million email messages spanning January through June 2017, obtained via public records request. 219,140 unique email addresses — 3,303 internal county staff and 215,837 external contacts. Analyzed in under 10 minutes using the Email Metadata Analytics Platform.

---

## Key Findings

### 1. The County's Email is 74% Machine-Generated

The top senders are not people — they are systems. Prosecutors' case notification alerts, police document management (OnBase), emergency paging (HipLink), building permit notifications, and Exchange infrastructure generate the vast majority of traffic. The Gini coefficient of 0.93 (where 1.0 = one sender sends everything) reveals extreme concentration: the top 10 senders account for 17.8% of all message traffic.

**Why it matters:** Understanding the machine-to-human ratio is critical for capacity planning, IT budgeting, and realistic workload assessment. A naive "email volume" analysis that doesn't separate automated traffic from human communication would dramatically overstate how busy staff actually are.

### 2. Five People Hold the Organization Together

Network analysis detected 628 distinct communication communities within the county. But five individuals — Jamie Burchett, M. Waitt, S. Akerlund, Simone Ramel-McKay, and County Clerk Kari Reardon — serve as critical bridges connecting otherwise isolated groups. Removing any one of them from the network would fragment inter-departmental communication.

**Why it matters:** These "organizational linchpins" represent a key-person risk. If one leaves, retires, or is out for an extended period, information flow between departments degrades. Succession planning should prioritize these roles.

### 3. Government Staff Reply in 22 Minutes

The median reply time across 60,484 detected reply pairs is just 22 minutes. 86.8% of conversational pairs respond within 4 hours. This is faster than most private-sector benchmarks and suggests a responsive organizational culture.

**Why it matters:** Response time metrics — derived entirely from timestamps and sender/recipient pairs, not email content — provide an objective measure of organizational responsiveness that surveys cannot.

### 4. After-Hours Email is Real, But Concentrated

18.7% of messages are sent outside business hours. 7.4% are sent on weekends. However, the majority of after-hours traffic is automated (system alerts, paging, monitoring). A handful of human staff show 60-75% after-hours email rates — potential indicators of workload imbalance, burnout risk, or roles that should be reclassified as shift work. The 911 Duty Supervisor at 57.9% after-hours is expected; a county clerk staff member at 68.6% is not.

**Why it matters:** After-hours analysis separates genuine overwork from automated noise. It identifies specific individuals who may need workload relief — something exit interviews catch too late.

### 5. The External Dependency Map is Revealing

The county's external communication reveals its operational dependencies: Washington State prosecutors (722K message edges), Spokane Police Department (a separate domain from the county), DSHS, defense attorney networks, King County, the Department of Justice, and Avista (the local utility). One external account — a Kittitas County prosecutor — generated 145,000 message edges into the Spokane County network, suggesting a major cross-jurisdictional caseload.

**Why it matters:** External communication patterns reveal inter-agency dependencies, vendor relationships, and cross-jurisdictional workload that don't appear in org charts or budget documents.

### 6. Automated Systems Leave Size Fingerprints

57,499 messages share the exact same byte size (4,198 bytes) — the HipLink emergency paging system. Another 42,570 share a second template size. Size forensics identifies auto-generated traffic without needing to read content, enabling accurate separation of human and machine email.

**Why it matters:** Size-based template detection is a reliable, content-free method for classifying email traffic. It works even when sender addresses are generic or shared.

---

## What Was NOT Accessed

This analysis used only four fields from each email header: **Date**, **Size**, **From**, and **To**. At no point was email body text, subject lines, or attachment content accessed, stored, or analyzed. All findings derive from communication patterns — who emailed whom, when, and how often.

---

## About the Platform

The Email Metadata Analytics Platform is a 20-page interactive dashboard that transforms raw email header data into organizational intelligence. It runs on-premise — your data never leaves your network. Connects directly to Microsoft 365 via the `Mail.ReadBasic.All` API scope, which is technically incapable of accessing email content.

Available for government agencies, law firms, HR consultants, and oversight bodies. Analysis engagements start at $10,000.

**Contact:** [your email/website here]
