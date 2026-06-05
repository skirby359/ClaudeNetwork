# What Data Is Accessed — Microsoft 365 Email Analysis

*Client-facing summary. Prepared by Tikor Consulting. This describes exactly what
information the analysis tool can and cannot access when connected to your
Microsoft 365 environment.*

---

## The short version

The analysis reads **email "envelope" metadata only** — the equivalent of the
*outside* of a sealed envelope: who sent it, who received it, and when. It
**cannot read the contents of any email**: no message bodies, no subject lines,
no attachments, no previews. This is not a policy promise — it is enforced
technically by Microsoft, at the permission level, and cannot be overridden by
the consultant.

---

## What we DO collect

For each message, the tool retrieves and stores:

| Field | Example | Purpose |
|---|---|---|
| **Date & time sent** | `2026-02-05 10:53` | Timing, volume, after-hours patterns |
| **Sender** (name + address) | `Carmela Conroy <carmela@…>` | Who initiated communication |
| **Recipients — To** (names + addresses) | `Field Data <fielddata@…>` | Who received it |
| **Recipients — Cc** (names + addresses) | — | Who was copied |
| **Recipient count** | `3` | Broadcast vs. one-to-one detection |

That is the complete list. From this, the tool builds aggregate patterns:
communication volume over time, who-talks-to-whom relationship maps, response
timing, and structural roles. **The analysis is about *patterns of
coordination*, not the substance of any conversation.**

## What we CANNOT collect

The connection uses a Microsoft permission called **`Mail.ReadBasic.All`**. By
Microsoft's own definition, this permission **excludes** the following, and the
service will refuse to return them even if asked:

- ❌ **Message body / text** — the actual content of any email
- ❌ **Body previews / snippets** — not even the first line
- ❌ **Attachments** — files, documents, images
- ❌ **Extended properties** — custom or hidden message fields

In addition, **by our own choice we do not even request the Subject line**,
which the permission would technically allow. We request strictly the
date/sender/recipient fields listed above and nothing more.

> **Two layers of protection.** (1) Microsoft technically blocks content access
> at the permission level. (2) On top of that, we narrow our request further than
> the permission allows, omitting subject lines. The consultant has no technical
> means to read what any email says.

---

## Addressing the concern: "Could the consultant see something they shouldn't?"

This is the right question to ask, so here is the honest, complete answer.

**What the consultant will see:** *that* communication happened, *between whom*,
and *when*. The relationship map — including the names and email addresses of
correspondents — is visible, because mapping coordination patterns is the
purpose of the engagement.

**What the consultant cannot see, under any circumstance:** what any email
*says*. No subject, no body, no attachment. This is not accessible to the tool,
to the consultant, or to anyone operating it.

**Two points worth your consideration:**

1. **Relationship metadata is itself information.** If the mere *existence* of
   correspondence with certain parties is sensitive (e.g., outside legal
   counsel, an HR investigation, a confidential third party), that *relationship*
   would appear in the data even though its *content* never could. If this is a
   concern, see the scoping and anonymization options below.

2. **Mailbox scope.** The `Mail.ReadBasic.All` permission is tenant-wide — it can
   read header metadata from **all mailboxes** in your Microsoft 365 tenant. If
   you wish to limit the analysis to specific mailboxes or departments, your
   Microsoft 365 administrator can apply an **Application Access Policy** that
   restricts the tool to a named set of mailboxes. We will work to whatever scope
   you define.

---

## Controls available to you

- **Mailbox scoping** — restrict the tool to specific mailboxes/departments via an
  Application Access Policy (configured by your admin).
- **Date-range scoping** — extract only a defined time window relevant to the
  engagement.
- **Display anonymization** — names and addresses can be masked in all reports and
  dashboards, so findings can be reviewed and shared without exposing individual
  identities.
- **Revocable access** — access runs through an app registration in *your* Azure
  directory. You grant it, and you can revoke it instantly at any time by removing
  consent or disabling the app. No credentials leave your control.
- **Local processing** — extracted metadata is processed within the engagement
  environment and is not transmitted to any third-party service.

---

## How access is granted and revoked

Access is established by your Microsoft 365 administrator registering an
application in **your** Azure Active Directory and granting it the
`Mail.ReadBasic.All` permission (and, if directory-wide mailbox discovery is
desired, `User.Read.All`). Both require a one-time **administrator approval**
("admin consent"); no individual user's mailbox can be accessed without your
administrator having authorized the application tenant-wide.

To revoke access at any time, your administrator removes the application's
permissions or deletes the app registration. Access ends immediately.

---

## Limiting access by date

The analysis can be confined to a specific time window, so that messages outside
the engagement period are **never retrieved from your tenant at all**.

- **Extraction-time date filter (primary control).** Every extraction can specify
  a start date (and, if needed, an end date). The tool requests only messages sent
  within that window using a server-side filter (`sentDateTime ge <start>`).
  Messages older than the start date are not returned by Microsoft and never reach
  the engagement environment — this is a filter applied *at the source*, not after
  the fact. For example, an engagement scoped to "January 2025 onward" will not
  retrieve, see, or store anything from December 2024 or earlier.
- **Engagement-window access (secondary control).** Because access depends on an
  app registration that you control, you can also bound access in *calendar* time:
  enable the app at the start of the engagement and disable or delete it at the
  end. Outside that window the tool cannot authenticate at all.
- **Client secret expiry.** The app's credential is issued with a fixed expiry
  date. Once it lapses, the tool can no longer connect until your administrator
  issues a new one — a natural backstop on how long access can persist.

We will agree the exact date range with you in writing before any extraction runs.

---

## Auditing what the application accessed

Because access flows through an app registration in **your** Microsoft 365
tenant, **you hold an independent audit trail** — you do not have to take our word
for what was accessed. Your administrator can review, at any time:

- **Entra sign-in logs (service principal sign-ins).** *Microsoft Entra admin
  center → Monitoring & health → Sign-in logs → "Service principal sign-ins" tab.*
  Every time the tool acquires a token is logged with timestamp, source IP, and
  success/failure. This shows *when* and *from where* the application connected.
- **Microsoft Graph activity logs (most granular).** *Entra admin center →
  Monitoring & health → Graph activity logs.* Records every individual Graph API
  request the application made — the exact URL, timestamp, and response. This lets
  you verify the tool called only the message-metadata endpoints and only the
  fields described in this document, and touched only the agreed mailboxes.
- **Microsoft Purview unified audit log (`MailItemsAccessed`).** *Microsoft
  Purview → Audit.* For tenants with the appropriate licensing, this logs
  application access to mailbox items, mailbox by mailbox — an additional,
  mailbox-owner-level record of access.
- **Permissions review.** *Entra admin center → Enterprise applications → [the
  app] → Permissions.* Shows exactly which permissions are consented
  (`Mail.ReadBasic.All`, and `User.Read.All` if enabled) — confirming no content,
  write, or send permissions were ever granted.

These logs are generated and retained by Microsoft under your control, are not
modifiable by the consultant, and can be exported for your compliance records. We
recommend your administrator enable diagnostic settings to retain sign-in and
Graph activity logs for the duration of the engagement.

---

*Questions about anything in this document are welcome. We are happy to walk your
IT and compliance teams through the exact technical configuration before any data
is accessed.*
