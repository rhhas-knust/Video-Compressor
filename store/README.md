# Publishing to the Microsoft Store

The app is published as a PWA (an installable web app). PWABuilder wraps the
GitHub Pages site in an MSIX package for the Store; the site itself stays the
single source, so every merge to `main` updates the installed app too.

Site: https://rhhas-knust.github.io/Video-Compressor/web/
Privacy policy: https://rhhas-knust.github.io/Video-Compressor/web/privacy.html
Screenshots (1920×1080): `web/screenshots/`

## 1. Partner Center account (free for individuals)
1. Go to https://storedeveloper.microsoft.com and choose **Individual**.
2. Sign in with a personal Microsoft account, verify your identity (ID + selfie)
   and fill in your details.

## 2. Reserve the app name
1. In Partner Center: **Apps and games → New product → MSIX or PWA app**.
2. Reserve a name. Plain "Video Compressor" is likely taken; try something
   distinctive (ideas below). The name is held for 3 months.
3. Open **Product management → Product identity** and copy three values:
   **Package/Identity/Name**, **Package/Identity/Publisher** (starts `CN=`) and
   **Package/Properties/PublisherDisplayName**.

## 3. Build the package with PWABuilder
1. Go to https://www.pwabuilder.com, enter
   `https://rhhas-knust.github.io/Video-Compressor/web/` and start.
2. Check the report: manifest, service worker and HTTPS should be green.
3. **Package for stores → Windows → Generate package**. Paste the three values
   from step 2.3 (Package ID, Publisher ID, Publisher display name) and use the
   reserved name. Version: `1.0.0` (raise it for every new submission).
4. Download the zip. It contains a `.msixbundle` (for the Store), a
   `.classic.appxbundle` (for older Windows 10) and install instructions for
   testing on your own PC first.

## 4. Submission in Partner Center
| Section | What to enter |
|---|---|
| Pricing and availability | Free; all markets (or choose) |
| Properties | Category **Photo & video**; privacy policy URL above; no extra hardware |
| Age ratings | Answer the questionnaire: no user content sharing, no ads, no purchases → usually rated 3+/Everyone |
| Packages | Upload the `.msixbundle` and the `.classic.appxbundle` |
| Store listings | Text below, the three screenshots, and the 300×300 logo PWABuilder generated |
| Submission options | Paste the certification notes below |

Then **Submit to the Store**. Certification usually takes from a few hours to
three business days; Partner Center emails you the result.

## Updating later
Changes merged to `main` reach users automatically (the app loads the site).
Only rebuild in PWABuilder and resubmit, with a higher version number, when the
name, icons or manifest change.

---

## Listing text (copy and paste)

**Name ideas:** Video Compressor & Clips · Squeeze Video Studio · ClipShrink ·
Video Compressor Offline

**Short description**
Make videos smaller without visible loss, pull out the audio, cut long videos
into Shorts and TikTok parts, and get a transcript with the key points. All on
your device.

**Description**
Video Compressor makes large video files smaller while keeping them looking the
same, and helps you reuse long videos for social media. Everything happens on
your own computer: your videos are never uploaded.

- Compress video: choose High, Balanced or Smallest and see the estimated size
  before you start. Uses your PC's hardware video encoder, so a long video
  finishes in minutes, not hours. Output plays everywhere (H.264) or is as small
  as possible (H.265/AV1).
- Extract audio: save a video's soundtrack as an .m4a file instantly, or make it
  smaller.
- Split for Shorts & TikTok: cut a long video into parts of 30 seconds to 10
  minutes, either instantly without losing quality or as vertical 9:16 clips with
  an optional "Part 1/N" label.
- Transcribe & notes: turn speech into a timestamped transcript and subtitles
  (.srt), with the important points picked out. Click any line to jump to that
  moment.
- Private: no account, no ads, no tracking. Works offline after the first start
  (transcription needs a one-time download of the speech model).

**Features (one per line)**
Compress videos without visible quality loss
Hardware-accelerated, many times faster than real time
Extract audio from video as .m4a
Split long videos for YouTube Shorts and TikTok
Vertical 9:16 clips with crop or blurred background
Transcripts, subtitles (.srt) and key-point notes
Runs on your device; videos are never uploaded

**Search terms (up to 7)**
video compressor, reduce video size, compress mp4, extract audio, shorts maker,
tiktok splitter, transcribe video

**Notes for certification**
This app processes videos locally in the browser engine (WebCodecs); nothing is
uploaded. To test: open any MP4 or MOV file, choose a tool and press the button
in step 3. The Transcribe & notes tool downloads a speech model (about 40 MB)
from huggingface.co the first time it is used, so it needs internet once. No
account or sign-in is required.
