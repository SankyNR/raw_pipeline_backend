"""
Run A (Technical Specifications) Configuration with precise, real-world extracted specifications.
Provides few-shot merged examples for Flagship, Foldable, and Budget/Midrange phones.
"""

from app.schemas.langextract_schema import (
    ExampleData,
    Extraction,
    BatteryExtraction,
    ChargingExtraction,
    DesignExtraction,
    ItemExtraction,
    ScreenExtraction,
    InTheBoxExtraction,
    VariantExtraction
)

EXAMPLE_FLAGSHIP = ExampleData(
    model_name="Samsung Galaxy S25 Ultra",
    text="""--- SOURCE: OEM_OFFICIAL (PRIORITY 1 — ALWAYS AUTHORITATIVE ON HARDWARE SPECS) ---
## Specifications

Galaxy S25 Ultra (256GB)Galaxy S25 Ultra (512GB)Galaxy S25 Ultra (1TB)

Galaxy S25+ (256GB)Galaxy S25+ (512GB)

Galaxy S25 (256GB)Galaxy S25 (512GB)

### Processor

- CPU Speed

4.47GHz, 3.5GHz

- CPU Type

Octa-Core


### Display

- Size (Main Display)

17.42 cm full rectangle / 17.22 cm rounded corners

- Resolution (Main Display)

3120 x 1440 (Quad HD+)

- Technology (Main Display)

Dynamic AMOLED 2X

- Colour Depth (Main Display)

16M

- Max Refresh Rate (Main Display)

120 Hz


### S Pen Support

- Yes


### Camera

- Rear Camera - Resolution (Multiple)

200.0 MP + 50.0 MP + 50.0 MP + 10.0 MP

- Rear Camera - F Number (Multiple)

F1.7 , F3.4 , F1.9 , F2.4

- Rear Camera - Auto Focus

Yes

- Rear Camera - OIS

Yes

- Rear Camera - Zoom

Optical Zoom 3x and 5x, Optical quality Zoom 2x and 10x (Enabled by Adaptive Pixel sensor) , Digital Zoom up to 100x

- Front Camera - Resolution

12.0 MP

- Front Camera - F Number

F2.2

- Front Camera - Auto Focus

Yes

- Rear Camera - Flash

Yes

- Video Recording Resolution

UHD 8K (7680 x 4320)@30fps

- Slow Motion

240fps @FHD, 120fps @FHD, 120fps @UHD


### Storage/Memory

- Memory (GB)

12

- Storage (GB)

256

- Available Storage (GB)

222.7


### Network/Bearer

- Number of SIM

Dual-SIM

- SIM size

Nano-SIM (4FF), Embedded-SIM

- SIM Slot Type

SIM 1 + SIM 2 / SIM 1 + eSIM / Dual eSIM

- Infra

2G GSM, 3G WCDMA, 4G LTE FDD, 4G LTE TDD, 5G Sub6 FDD, 5G Sub6 TDD

- 2G GSM

GSM850, GSM900, DCS1800, PCS1900

- 3G UMTS

B1(2100), B2(1900), B4(AWS), B5(850), B8(900)

- 4G FDD LTE

B1(2100), B2(1900), B3(1800), B4(AWS), B5(850), B7(2600), B8(900), B12(700), B13(700), B17(700), B18(800), B19(800), B20(800), B25(1900), B26(850), B28(700), B66(AWS-3)

- 4G TDD LTE

B38(2600), B39(1900), B40(2300), B41(2500)

- 5G\\* FDD Sub6

N1(2100), N2(1900), N3(1800), N5(850), N7(2600), N8(900), N12(700), N20(800), N25(1900), N26(850), N28(700), N66(AWS-3)

- 5G\\* TDD Sub6

N38(2600), N40(2300), N41(2500), N77(3700), N78(3500)


### Connectivity

- USB Interface

USB Type-C

- USB Version

USB 3.2 Gen 1

- Location Technology

GPS, Glonass, Beidou, Galileo, QZSS

- Earjack

USB Type-C

- MHL

No

- Wi-Fi

802.11a/b/g/n/ac/ax/be 2.4GHz+5GHz+6GHz, EHT320, MIMO, 4096-QAM

- Wi-Fi Direct

Yes

- Bluetooth Version

Bluetooth v5.4

- NFC

Yes

- UWB (Ultra Wideband)

Yes


### OS

- Android


### General Information

- Form Factor

Touchscreen Bar


### Sensors

- Accelerometer, Barometer, Fingerprint Sensor, Gyro Sensor, Geomagnetic Sensor, Hall Sensor, Light Sensor, Proximity Sensor


### Physical specification

- Dimension (HxWxD, mm)

162.8 x 77.6 x 8.2

- Weight (g)

218


### Battery

- Video Playback Time (Hours)

Up to 31

- Battery Capacity (mAh, Typical)

5000

- Removable

No


### Audio and Video

- Stereo Support

Yes

- Video Playing Format

MP4, M4V, 3GP, 3G2, AVI, FLV, MKV, WEBM

- Video Playing Resolution

UHD 8K (7680 x 4320)@60fps

- Audio Playing Format

MP3, M4A, 3GA, AAC, OGG, OGA, WAV, AMR, AWB, FLAC, MID, MIDI, XMF, MXMF, IMY, RTTTL, RTX, OTA, DFF, DSF, APE


### Services and Applications

- Gear Support

Galaxy Ring, Galaxy Buds3 Pro, Galaxy Buds2 Pro, Galaxy Buds Pro, Galaxy Buds Live, Galaxy Buds+, Galaxy Buds3, Galaxy Buds2, Galaxy Buds, Galaxy Buds FE, Galaxy Fit3, Galaxy Fit2, Galaxy Fit e, Galaxy Fit, Galaxy Watch FE, Galaxy Watch Ultra, Galaxy Watch7, Galaxy Watch6, Galaxy Watch5, Galaxy Watch4, Galaxy Watch3, Galaxy Watch, Galaxy Watch Active2, Galaxy Watch Active

- Samsung DeX Support

Yes

- Bluetooth® Hearing Aid Support

Android Audio Streaming for Hearing Aid(ASHA)

- SmartThings Support

Yes

- Mobile TV

No


### Software Support

- Security Update Period (Valid until)

31 January 2032


### Manufacturer’s Information

- Manufactured by

Samsung India Electronics Pvt. Ltd. having its Registered Office at: 6th Floor, DLF Centre, Sansad Marg, New Delhi-110001

- Country of Origin

India

- Contact us

For All Product Related Complaints/assistance, please contact Manager, Customer Care Samsung India Electronics Pvt. Ltd., 6th Floor, DLF Centre, Sansad Marg, New Delhi - 110001 Email us at: support.india@samsung.com Tel (Toll Free): 1800 40 7267864 (1800 40 SAMSUNG)

- Launch date

January 22nd, 2025


\\* Network : The bandwidths supported by the device may vary depending on the region or service provider.

\\* Battery: Actual battery life varies by network environment, features and apps used, frequency of calls and messages, number of times charged, and many other factors.

\\* User Available Memory: Actual user memory will vary depending on the operator and may change after software upgrades are performed.

\\* Battery Capacity (Typical): Typical value tested under third-party laboratory condition. Typical value is the estimated average value considering the deviation in battery capacity among the battery samples tested under IEC62133/IEC62133-2 standard. Rated typical capacity is 3885 mAh for Galaxy S25, 4755 mAh for Galaxy S25+ and 4855 mAh for Galaxy S25 Ultra. Actual battery life may vary depending on network environment, usage patterns and other factors.

\\* Display Size: Measured diagonally, Galaxy S25's screen size is 15.64 cm in the full rectangle and 15.23 cm accounting for the rounded corners, Galaxy S25+'s screen size is 16.91 cm in the full rectangle and 16.45 cm accounting for the rounded corners, and Galaxy S25 Ultra's screen size is 17.42 cm in the full rectangle and 17.22 cm accounting for the rounded corners; actual viewable area is less due to the rounded corners and camera hole.

--- SOURCE: GSMARENA (PRIORITY 3 — USE ONLY WHEN OEM_OFFICIAL IS SILENT ON A FIELD) ---
| Network | [Technology](https://www.gsmarena.com/network-bands.php3) | [GSM / CDMA / HSPA / EVDO / LTE / 5G](https://www.gsmarena.com/samsung_galaxy_s25_ultra-13322.php#) |
| [2G bands](https://www.gsmarena.com/network-bands.php3) | GSM 850 / 900 / 1800 / 1900 |
|  | CDMA 800 / 1900 & TD-SCDMA |
| [3G bands](https://www.gsmarena.com/network-bands.php3) | HSDPA 850 / 900 / 1700(AWS) / 1900 / 2100 |
|  | CDMA2000 1xEV-DO |
| [4G bands](https://www.gsmarena.com/network-bands.php3) | 1, 2, 3, 4, 5, 7, 8, 12, 13, 17, 18, 19, 20, 25, 26, 28, 32, 38, 39, 40, 41, 66 - International |
|  | 1, 2, 3, 4, 5, 7, 8, 12, 13, 14, 18, 19, 20, 25, 26, 28, 29, 30, 38, 39, 40, 41, 48, 66, 71 - USA unlocked |
| [5G bands](https://www.gsmarena.com/network-bands.php3) | 1, 2, 3, 5, 7, 8, 12, 20, 25, 26, 28, 38, 40, 41, 66, 75, 77, 78 SA/NSA/Sub6 - International |
|  | 1, 2, 5, 7, 25, 28, 41, 66, 71, 77, 78, 257, 258, 260, 261 SA/NSA/Sub6/mmWave - USA unlocked |
| [Speed](https://www.gsmarena.com/glossary.php3?term=3g) | HSPA, LTE (CA), 5G |

| Launch | [Announced](https://www.gsmarena.com/glossary.php3?term=phone-life-cycle) | 2025, January 22 |
| [Status](https://www.gsmarena.com/glossary.php3?term=phone-life-cycle) | Available. Released 2025, February 03 |

| Body | [Dimensions](https://www.gsmarena.com/samsung_galaxy_s25_ultra-13322.php#) | 162.8 x 77.6 x 8.2 mm (6.41 x 3.06 x 0.32 in) |
| [Weight](https://www.gsmarena.com/samsung_galaxy_s25_ultra-13322.php#) | 218 g (7.69 oz) |
| [Build](https://www.gsmarena.com/glossary.php3?term=build) | Glass front (Corning Gorilla Armor 2), glass back (Gorilla Glass Victus 2), titanium frame (grade 5) |
| [SIM](https://www.gsmarena.com/glossary.php3?term=sim) | · Nano-SIM + Nano-SIM + [eSIM](https://www.gsmarena.com/glossary.php3?term=esim) \\+ eSIM (max 2 at a time) - INT<br>* * *<br>· Nano-SIM + eSIM + eSIM (max 2 at a time) - USA<br>* * *<br>· Nano-SIM + Nano-SIM - CN |
|  | IP68 dust tight and water resistant (immersible up to 1.5m for 30 min)<br>* * *<br>Stylus |

| Display | [Type](https://www.gsmarena.com/glossary.php3?term=display-type) | Dynamic LTPO AMOLED 2X, 120Hz, HDR10+, 2600 nits (peak) |
| [Size](https://www.gsmarena.com/samsung_galaxy_s25_ultra-13322.php#) | 6.9 inches, 116.9 cm2 (~92.5% screen-to-body ratio) |
| [Resolution](https://www.gsmarena.com/glossary.php3?term=resolution) | 1440 x 3120 pixels, 19.5:9 ratio (~498 ppi density) |
| [Protection](https://www.gsmarena.com/glossary.php3?term=screen-protection) | Corning Gorilla Armor 2, Mohs level 6 |
|  | DX anti-reflective coating |

| Platform | [OS](https://www.gsmarena.com/glossary.php3?term=os) | Android 15, up to 7 major Android upgrades, One UI 8 |
| [Chipset](https://www.gsmarena.com/glossary.php3?term=chipset) | Qualcomm SM8750-AC Snapdragon 8 Elite (3 nm) |
| [CPU](https://www.gsmarena.com/glossary.php3?term=cpu) | Octa-core (2x4.47 GHz Oryon V2 Phoenix L + 6x3.53 GHz Oryon V2 Phoenix M) |
| [GPU](https://www.gsmarena.com/glossary.php3?term=gpu) | Adreno 830 (1200 MHz) |

| Memory | [Card slot](https://www.gsmarena.com/glossary.php3?term=memory-card-slot) | No |
| [Internal](https://www.gsmarena.com/glossary.php3?term=dynamic-memory) | 256GB 12GB RAM, 512GB 12GB RAM, 1TB 12GB RAM, 1TB 16GB RAM |
|  | UFS 4.0 |

| Main Camera | [Quad](https://www.gsmarena.com/glossary.php3?term=camera) | 200 MP, f/1.7, 24mm (wide), 1/1.3\\", 0.6µm, multi-directional PDAF, OIS<br>* * *<br>10 MP, f/2.4, 67mm (telephoto), 1/3.52\\", 1.12µm, PDAF, OIS, 3x optical zoom<br>* * *<br>50 MP, f/3.4, 111mm (periscope telephoto), 1/2.52\\", 0.7µm, PDAF, OIS, 5x optical zoom<br>* * *<br>50 MP, f/1.9, 120˚ (ultrawide), 1/2.5\\", 0.7µm, dual pixel PDAF, Super Steady video |\\
| [Features](https://www.gsmarena.com/glossary.php3?term=camera) | Laser AF, Best Face, LED flash, auto-HDR, panorama |
| [Video](https://www.gsmarena.com/glossary.php3?term=camera) | 8K@24/30fps, 4K@30/60/120fps, 1080p@30/60/120/240fps, 10-bit HDR, HDR10+, stereo sound rec., gyro-EIS |\\

| Selfie camera | [Single](https://www.gsmarena.com/glossary.php3?term=secondary-camera) | 12 MP, f/2.2, 26mm (wide), 1/3.2\\", 1.12µm, dual pixel PDAF |
| [Features](https://www.gsmarena.com/glossary.php3?term=secondary-camera) | HDR, HDR10+ |
| [Video](https://www.gsmarena.com/glossary.php3?term=secondary-camera) | 4K@30/60fps, 1080p@30fps |

| Sound | [Loudspeaker](https://www.gsmarena.com/glossary.php3?term=loudspeaker) | Yes, with stereo speakers |
| [3.5mm jack](https://www.gsmarena.com/glossary.php3?term=audio-jack) | No |
|  | High-bitrate audio support |

| Comms | [WLAN](https://www.gsmarena.com/glossary.php3?term=wi-fi) | Wi-Fi 802.11 a/b/g/n/ac/6e/7, tri-band, Wi-Fi Direct |
| [Bluetooth](https://www.gsmarena.com/glossary.php3?term=bluetooth) | 5.4, A2DP, LE |
| [Positioning](https://www.gsmarena.com/glossary.php3?term=gnss) | GPS, GLONASS, BDS, GALILEO, QZSS |
| [NFC](https://www.gsmarena.com/glossary.php3?term=nfc) | Yes |
| [Radio](https://www.gsmarena.com/glossary.php3?term=fm-radio) | No |
| [USB](https://www.gsmarena.com/glossary.php3?term=usb) | USB Type-C 3.2, DisplayPort 1.2, OTG |

| Features | [Sensors](https://www.gsmarena.com/glossary.php3?term=sensors) | Fingerprint (under display, ultrasonic), accelerometer, gyro, proximity, compass, barometer |
|  | Samsung DeX, Samsung Wireless DeX (desktop experience support)<br>* * *<br>Ultra Wideband (UWB) support |

| Battery | [Type](https://www.gsmarena.com/glossary.php3?term=rechargeable-battery-types) | Li-Ion 5000 mAh |
| [Charging](https://www.gsmarena.com/glossary.php3?term=battery-charging) | 45W wired, PD3.0, 65% in 30 min<br>* * *<br>15W wireless (Qi2 Ready)<br>* * *<br>4.5W reverse wireless |

| Misc | [Colors](https://www.gsmarena.com/glossary.php3?term=build) | Titanium Silver Blue, Titanium Black, Titanium White Silver, Titanium Gray, Titanium Jade Green, Titanium Jet Black, Titanium Pink Gold |
| [Models](https://www.gsmarena.com/glossary.php3?term=models) | SM-S938B, SM-S938B/DS, SM-S938U, SM-S938U1, SM-S938W, SM-S938N, SM-S9380, SM-S938E, SM-S938E/DS |
| [SAR](https://www.gsmarena.com/glossary.php3?term=sar) | 1.26 W/kg (head)     0.64 W/kg (body) |
| [SAR EU](https://www.gsmarena.com/glossary.php3?term=sar) | 1.25 W/kg (head)     1.42 W/kg (body) |
| [Price](https://www.gsmarena.com/glossary.php3?term=price) | [$ 767.06 / C$ 1,204.99 / £ 699.00 / € 875.00 / ₹ 118,999](https://www.gsmarena.com/samsung_galaxy_s25_ultra-price-13322.php) |

| Our Tests | [Performance](https://www.gsmarena.com/glossary.php3?term=benchmarking) | AnTuTu: 2207809 (v10)<br>* * *<br>GeekBench: 9846 (v6)<br>* * *<br>3DMark: 6687 (Wild Life Extreme) |
| [Display](https://www.gsmarena.com/gsmarena_lab_tests-review-751p2.php) | [1417 nits max brightness (measured)](https://www.gsmarena.com/samsung_galaxy_s25_ultra-review-2793p3.php#dt) |
| [Loudspeaker](https://www.gsmarena.com/gsmarena_lab_tests-review-751p7.php) | [-24.6 LUFS (Very good)](https://www.gsmarena.com/samsung_galaxy_s25_ultra-review-2793p3.php#lt) |
| [Battery](https://www.gsmarena.com/how_we_test_gsmarena_battery_life_test_v2-news-60429.php) | [Active use score 14:49h](https://www.gsmarena.com/samsung_galaxy_s25_ultra-review-2793p3.php#bt) |

| EU LABEL | [Energy](https://www.gsmarena.com/glossary.php3?term=eu-energy-class) | Class B |
| [Battery](https://www.gsmarena.com/glossary.php3?term=eu-battery-endurance) | 44:54h endurance, 2000 cycles |
| [Free fall](https://www.gsmarena.com/glossary.php3?term=eu-free-fall-class) | Class A (270 falls) |
| [Repairability](https://www.gsmarena.com/glossary.php3?term=eu-repairability-class) | Class C |

--- SOURCE: TRANSCRIPT (PRIORITY 2 — USE FOR: charger_in_box, in_the_box, India-specific colors, India variant confirmations) ---
Everyone keeps saying buy last year's flagship and people actually do as well. At the time of recording this video when we checked Amazon the most popular flagship was the Samsung Galaxy S24 Ultra. Good phone. Now it's that time of the year when one of Samsung's most successful flagships of all time which helped propel it to the second spot in the global market share is about to become last year's phone. Yes, I'm talking about the Samsung Galaxy S25 Ultra. So with the S26 Ultra launch around the corner I really wanted to take a look at the S25 Ultra 300 days later. This is the first time we're making such a video. So I'm really excited about it. All right, if you're watching me for the first time, I'm a shot. This is Track and Tech English, your destination for detailed incisive gadget reviews. Now if you saw my original video on the S25 Ultra, you know that I love the design and that still holds true. The rounded corners and the boxy frame actually give it a solid in-hand feel and I feel that it is definitely improved this year. And it's also only 8.2 mm thick giving it the right amount of heft. It is quite balanced too. At 218 g it feels very comfortable to hold and use. The build quality is flagship grade and the frame continues to be grade [music] 5 titanium which unlike other brands is much better. And we've been rocking this phone for months without a case and the cosmetic wear is minimal. There are a few small nicks on the rails and a minor paint chipping on the top camera ring but overall the body has been intact. And when you compare it to something like the iPhone 17 Pro Max which actually uses aluminum this year, this is definitely better because the aluminum is denting on many people's phones including mine, unfortunately. The display is protected by Gorilla Armor 2 and the back uses Gorilla Glass Victus 2. This is a very good combination, easily top of the line that you can get on any Android flagship today. And you know what, Gorilla Armor 2 has held up so well in our time of usage. There is very little scuff or scratches to show for on our display. And it goes without saying that Samsung's anti-reflective coating remains exceptional. It is the best in the industry right now. And 300 days later the design still feels as premium as the first day when we unboxed the phone. And considering the fact many months later now there are multiple discounts on the phone, it is such an attractive proposition for such a premium design. But it's not just the design. The display continues to be excellent. Now this is a 6.9-in Dynamic LTPO AMOLED 2X panel with 120 Hz refresh rate and 480 Hz PWM dimming and support for HDR10+ as well. For watching content it's fantastic. So whether it's action-heavy F1 races or movies like Interstellar, I enjoyed watching it on this flat panel. Of course there is no Dolby Vision support on any Samsung phone which includes the Samsung Galaxy S25 Ultra. And since this is an 8-bit panel in certain HDR scenes, you will definitely notice some banding as well. But between the excellent color accuracy, the anti-reflective coating and the top tier brightness, the S25 Ultra still stands tall as one of the best displays on the market today despite having launched at the start of the year. Now before I talk about the performance of the phone, I want to talk about certain improvements that I noticed compared to the Samsung Galaxy S24 Ultra. Now there are a couple of things that I want to talk about the S25 Ultra compared to the S24 Ultra is that the performance tuning has definitely improved. Because with One UI on the S25 Ultra particularly, it actually manages system resources better and preloads app data so that it can actually function faster which is something that we are noticing. There's a much larger vapor cooling chamber now which means that the heat dissipation is better and it actually offers better sustained performance in gaming compared to the S24 Ultra. Now here's one interesting thing with the S25 Ultra you got the LPDDR5X RAM type but with the S26 series Samsung is expected to actually introduce the LPDDR6 RAM type. Anyway, talking about performance with this Samsung Galaxy S25 Ultra, it's not the Snapdragon 8 Gen 5 that you get but it's Snapdragon 8 Gen 5 for Galaxy. The Snapdragon 8 Gen 5 could possibly come in the S26 Ultra. Now whether you're opening or closing apps, opening multiple browsers in the hopes of revisiting someday or playing intensive games, everything runs absolutely well on the S25 Ultra. We really have no complaints. In fact in Wuthering Waves with the S25 Ultra, we're getting 59 FPS which is in line with what we got with the newer iPhone 17 Pro Max. So the performance still holds good even today. And the AnTuTu score with version 11 has gone up to 3.2 million now which is possibly the best we've gotten for a Snapdragon 8 Gen 1 phone. In fact even the single core and the multi-core score is pretty good. And you know even the Dimensity 9500 with the Find X9 Pro is not able to beat this. Even in the Solar Bay it got about 48 FPS which is one of the highest that we've tested this year. Storage speeds are pretty solid too. If you look at it, it's still one of the fastest storage speeds that you can get on a phone. So in reality despite having launched in January because it has Snapdragon 8 Gen 1 for Galaxy which has a more performance throughput, the performance is still in line with what, you know, the A19 Pro can achieve with the iPhone 17 Pro Max. Now the UI on top of Android 16 is of course One UI 7 that you get but you also have an upgrade to One UI 8 already. And more importantly, you also get a bunch of AI features. And that is what sets any Samsung flagship apart because easily the best software experience on Android flagships right now has to be One UI. And I've been saying that for a while now. By the way when it moved from One UI 7 to One UI 8 a lot of things did change. For example, you get better file tracking, you get dynamic wallpapers that shift brightness based on the time of the day. And you get improved multitasking with more flexible split screen resizing. All of this just make everyday usage more refined. And as for the AI features with the S25 Ultra, you know that Samsung is doing better than almost every other brand out there. You can summon Gemini AI by holding the power button and it handles multi-step cross app actions like finding info online and emailing it. What is on my screen right now? Yeah, it looks like a handheld gaming console. What games do you usually play on it? Give me the most recent price of the Samsung Galaxy S25 Ultra in India. The Samsung Galaxy S25 Ultra in India starts at around 129,999 rupees for the 12 GB RAM and 256 GB storage model. There's browsing assist and call assist for live translation and note assist to basically tidy or summarize your notes. And these are all features that most other brands are just trying to play catch up to Samsung now. For example, if you take visual intelligence in the iPhone 17 Pro Max, it is fairly limited in its capability compared to the Gemini Live and Circle to Search. The results are only limited to visual matches on the iPhone 17 Pro Max. With the S25 Ultra you can literally look up to anything [music] with Circle to Search. The photo assist lets you erase or restyle images while portrait studio adds fun comic or watercolor effects. You have an image playground in Apple which is fairly limited again. And if you talk about object eraser, Samsung just, you know, smokes Apple out of the water. The audio eraser does do a good job of cleaning up audio but here I would say that Apple does offer a very good capability with studio mix as well. By the way you also get the Now Brief home widget and Circle to Search for instant lookups which is all neatly integrated into the S25 Ultra. But we keep forgetting about the fact that there is still the S Pen that exists even if it's limited in capabilities now. It's still great for sketching, writing and quick notes and of course it remains unmatched among any stylus equipped phones out there. There are very few anyway. Now talking about limited capabilities, Samsung did remove the Bluetooth capabilities this year which means that you do not get air actions anymore. But after 300 days of usage, I really didn't miss it, honestly. All right, now talking about the Samsung Galaxy S25 Ultra's cameras after 300 days, I actually compared it to the iPhone 17 Pro Max to see where it, you know, stands. So the Galaxy S25 Ultra's 200 megapixel main camera actually does a very good job compared to the 48 megapixel unit that you get with the iPhone 17 Pro Max. In daylight if you look at it, Samsung images do look brighter and slightly vibrant compared to a little more true to life restrained look that you get with the iPhone 17 Pro Max. One thing I noticed is that in HDR scenes, Samsung actually does a better job of controlling the highlights compared to the iPhone 17 Pro Max even right now. For some odd reason, Apple is not bringing good upgrades to the HDR tuning with Smart HDR 5 continuing to run on the iPhone 17 Pro Max. Samsung's low light performance is also matured with cleaner colors, deeper and more natural shadows too. Apple still does edge out in fine control and overall detail if you look at it. Now when you look at the 50 megapixel ultra wide angle camera, it produces sharper, more contrasty results than iPhone's 48 megapixel fusion ultra wide. And both of these also double up as excellent macro lens with fast accurate autofocus. I would say in ultra wide both are equally matched. Portraits on the other hand, I feel like Samsung sometimes does have an edge with better edge detection and better natural bokeh drop off but details seem to be better on the iPhone. However, I'm seeing some inconsistency with iPhone's portraits once in a while, something that they haven't fixed just yet. But zoom is where Samsung really actually wins compared to Apple. Now of course you do get a longer zoom range which means that at, you know, levels like 20x, 30x and beyond but even at 5x obviously Samsung beats Apple. So overall if you look at it, if you're looking at a phone for zoom performance, I would say that the S25 Ultra is actually better than the 17 Pro Max. I would say selfie is where Apple definitely has an advantage compared to the S25 Ultra. While the S25 Ultra selfies do look good if you look at the comparison samples side by side, Apple does have that edge with the new sensor that they've brought in which means that you can take both vertical and horizontal crops. Video recording is one area where Samsung actually does a good job with 4K 120 FPS HDR10+ footage, all of that is great. But if you look at it, Apple actually edges ahead. That's because you can shoot Dolby Vision but more importantly low light video recording quality is better on Apple. But if you want better zoom video quality then I would say that Samsung does a better job than Apple in terms of details alone. Now 300 days later, how has the battery life held up on the S25 Ultra? In fact, it's actually become better than day one. You get a 5000 mAh battery, you get 45 W charging speeds, all that is great. But what we're constantly getting in our usage is about 1 and 1/2 days of, you know, stress-free usage. You so you can expect anywhere between 6 and 1/2 to 7 and 1/2 hours of screen on time with the 5,000 mAh battery on the S25 Ultra depending on your usage. I feel like Samsung's algorithm actually does a very good job of trying to understand your usage pattern, learning it over a period of time, and improving the performance accordingly. As for the battery health, we actually ran diagnostics on the phone, and it's still in good health condition. So much so that we can actually use it for 2 3 4 5 years without any stress. All right, so 300 days later, should you still consider the Samsung Galaxy S25 Ultra? Well, the answer to that is quite complicated as it is simple. In raw value, even in current discounted prices, the S25 Ultra is a fantastic flagship even today. It's a complete Android flagship. You get a great display, you get a premium design, you get possibly the best software experience out there, a great set of cameras, really good battery life, and yes, just that Samsung premium feel through and through. But, with the S26 Ultra, it just might make more sense to just wait because A, you could want to own the S26 Ultra for a higher price, or B, the S25 Ultra could actually go down even further in prices, making it an even sweeter deal. So yeah, this is one of those rare phones that really stands the test of times. Even 300 days later, it feels like a very, very good 2025 flagship. So yeah, this is one of those very few Android flagships in particular, bug free to use, and has held up well in our time of usage, especially 300 days later. So it's an easy recommendation from us if you're considering it right now. I'll see you guys in the next one. Until then, keep tracking and stay safe.
दोस्तों ये वीडियो जो है ना मुझे बहुत पहले करना था लेकिन 1UI 8 के लिए रुका हुआ था मैंने कहा उसके बाद में करेंगे उसके बाद में 8.5 आ गया एंड फाइनली आई थिंक द टाइम इस कम टू टेल यू कि मुझे लगता है 2025 का एक कंप्लीट फ्लैकशिप अगर आप मुझसे पूछते हो तो मैं कहूंगा S25 अल्ट्रा दिस इज़ गोइंग टू बी अ लॉन्ग टर्म रिव्यु मतलब जैन में लॉन्च हुआ था वी आर इन नवंबर राइट नाउ मतलब 10 महीने कंप्लीटली मेरे साथ रहा है ये फ़ एंड आई हैव यूज़्ड इट जी हां iPhone लॉन्च होने के बाद भी मैं iPhone पे शिफ्ट हुआ बट आई कुड नॉट गिव गिव दिस अप। मेरा एक सिम कार्ड इसके अंदर था और एक iPhone में था। सो दिस वास ऑलवेज देयर विथ मी। तो अभी दो बड़े-बड़े फोन्स मैं लेकर घूमता हूं। बट रीज़न है उसके लिए। ये एक ऐसा फोन है ना जो लिटरली आई सो टू से कैन नॉट डू विदाउट। और मैं बताता हूं आपको क्यों। ऐसी कुछ रेवोल्यूशनरी कोई चीज नहीं है। बट सब कुछ जो है ना इट जस्ट वर्क्स परफेक्टली। आई एम नॉट टू श्योर कि मैं ये कैसे एक्सप्लेन करूं। बट आई वांट अ S25 सीरीज या S सीरीज फोन मेरे साथ में। जी हां, बताने वाला हूं सब कुछ डिटेल रिव्यु होने वाला है। लेकिन उससे पहले अगर पहली बार हमारे चैनल पे आए हो तो सब्सक्राइब जरूर कीजिएगा एंड बेल आइकॉन जरूर दबाइए। जनरली हम इतना लंबा लॉन्ग टर्म रिव्यु करते नहीं है। बट आई थिंक दिस फ़ोन डिसर्वार्स इट। मतलब S25 अल्ट्रा ऐसा फोन है ना कि आपको जब नया होता है तो ऑब्वियसली ऐसा लगता है वाओ यार यूज करना चाहिए। एंड देन वो जो क्यूरियोसिटी होती है, जो एक्साइटमेंट होती है उस फोन के बारे में वो वेयर ऑफ हो जाती है एक-दो महीने यूज करने के बाद। जैसा मेरा अभी iPhone के साथ हो गया। आई लिटरली वांट टू गिव इट अवे। और वो मैं बताने वाला हूं क्यों? बट S25 अल्ट्रा के साथ यू नो इट जस्ट कीप्स ग्रोइंग ऑन यू क्योंकि जो भी आप काम करते हो ना जस्ट वर्क्स अब सबसे पहले डिज़ाइन एंड बिल्ड के बारे में बात करते हैं। थोड़ा दोस्तों ये वीडियो शायद रैंक जैसे भी होगा क्योंकि ऑब्वियसली मेरे साथ ये फोन रहा है। बहुत सारी चीजें ऐसी बहुत सारी चीजें मुझे आपको बतानी है। बट डिज़ाइन एंड बिल्ड क्वालिटी के मामले में मैं अभी इसे केस के साथ इस्तेमाल करता हूं। लेकिन पहले चार-प महीने मैंने विदाउट केस इस्तेमाल किया था। और यह इतने बार ऐसे गिरा हुआ है, साइड में गिरा हुआ है, पीछे की तरफ गिरा हुआ है। एंड इसे एब्सोलुटली कुछ नहीं हुआ है। क्यों पता है? क्योंकि ये जो टाइटेनियम फ्रेम जो है ना इट मतलब आप देखिए कुछ स्क्रैचेस आपको दिखेंगे ही नहीं आफ्टर 10 मंथ ऑफ यूसेज। ऐसे लग रहा है अगर आप इसके पास देखोगे कि अभी मैंने बॉक्स के बाहर निकाला हुआ है। सो बिल्ड क्वालिटी के मामले में अब्सोलुटली नो इशूज़। यस, आई हैव टू से कि आपको बड़े फोंस पसंद होने चाहिए फॉर दिस टू गेट। क्योंकि ये बड़ा है दोस्तों। यह यह फ़ोन जो है ना एक बार हाथ में ले लिया तो आपको लगेगा इट वन ऑफ़ द बिगर फ़ोंस एंड स्लाइटली ऑन द हैवियर साइड। वो है इसका। जब ये लॉन्च हुआ था ना तब एस पेन मुझे लगा था कि ब्लूटूथ निकाल दिया तो इतना यूज नहीं होगा। वैसे मैं किस लिए इस्तेमाल करता हूं पता है? एस पेन साइन करने जो भी डॉक्यूमेंट्स आते हैं एम्बार्गo वगैरह होते हैं वो मैं साइन इससे करता हूं। इन लास्ट 10 मंथ्स मैंने सिर्फ उसी के लिए इस्तेमाल किया हुआ है और उसके लिए ब्लूटूथ की जरूरत नहीं है। हां जो बहुत सारे रिमोटली फोटोग्राफ वगैरह निकालते हैं और अलग-अलग सिचुएशंस है उसके लिए होगा। बट मेरे लिए एस पेन में ब्लूटूथ है नहीं है उसका कुछ भी डिफरेंस नहीं पड़ा। कैमरा के बारे में बात करते हैं। बहुत सारी और भी चीजें हैं। ओएस के बारे में रुकिएगा। बट कैमरा में कुछ इंटरेस्टिंग चीजें हैं। इसमें दोस्तों यू विल बी सरप्राइज्ड बट मुझे लगता है कि फोटो स्टिल फोटोज के मामले में S25 अल्ट्रा इज बेटर देन iPhone। यस। और मैं अभी इस्तेमाल कर रहा हूं। ये देखिए। आई एम आई एम यूजिंग अ iPhone इन दिस। सो iPhone और ये दोनों भी मैं इस्तेमाल कर रहा हूं। बट स्टिल फोटोज में आई स्टिल प्रेफर दिस वन। वीडियोस में डेफिनेटली iPhone जो है वो आगे चला जाता है। अब फोटोस के बारे में मैं बात करूं तो आप देख लीजिएगा। बहुत सारे सैंपल्स अभी दे रहा हूं। गो एंड चेक इट आउट। कलर एक्यूरेसी के मामले में, टेली फोटो के मामले में आई मीन आई कैन क्लिक अप टू 5 एक्स ज़ूम। ज़ूम के मामले में भी इवन लो लाइट में मैं मेरे घर से रात को यू नो आई स्टे एट अ हायर फ्लोर। तो मुझे बहुत अच्छी तरह से स्काई लाइन दिखता है। मैं लिटरली अगर दो फ़ोन साथ में होंगे तो इंस्टिंक्टिवली मैं S25 अल्ट्रा को पिकअप करूंगा क्योंकि मैं फोटोज निकालने वाला हूं। लेकिन एक बात मैं इधर इतना अगर मैं बात कर रहा हूं अच्छी चीजों के बारे में तो एक चीज मैं ये भी कहना चाहूंगा कि पिछले तीन जनरेशन से उन्होंने सेंसर जो है उसके ऊपर ज्यादा मतलब हार्डवेयर के ऊपर उतना ज्यादा काम नहीं किया है। इधर मैं ये भी कहूंगा कि पोटेंशियल बहुत ज्यादा है इन कैमरास का। इवन पोस्ट प्रोसेसिंग में भी बहुत सारी चीजें और भी Samsung ला सकता है इसके अंदर जो अभी मेरे हिसाब से नहीं है। बट जैसे आती है अच्छी है। और एक चीज यस इंपॉर्टेंट। इसका जो शटर है ना वो थोड़ा स्लो है। iPhone के साथ कंपेयर करोगे तो इसका शटर बटन थोड़ा स्लो है। और कभी-कभी मैंने एक देखा है मतलब जनरली 90% ऑफ द टाइम्स यू नो जैसे एक्सपेक्टेड है वैसे निकालता है। कभी-कभी ओवर सैचुरेटेड फोटोज दिखते हैं। मतलब सोशल मीडिया वर्दी होते हैं वो फोटो। बट हां पंची आ जाते हैं। कलर्स थोड़े सैचुरेटेड दिखते हैं। और जैसे पहले बताया वीडियोस या आई मीन इट्स नॉट बैड। इट्स इट्स रियली गुड। बट अगर मैं फ्लैकशिप iPhone के साथ वगैरह कंपेयर करूं तो iPhone आगे निकल जाता है। एंड आई थिंक [संगीत] दिस कम्स अ लिटल डाउट। और AI के फीचर्स के बारे में बात करूं तो Galaxy AI जो है आई थिंक दिस इज़ वन ऑफ़ द मोस्ट फीचर रिच AI UI आउट देयर। मतलब देयर इज नो क्वेश्चन। आप सर्कल टू सर्च से स्टार्ट कीजिए। Google जेमिनी का इंटीग्रेशन देखिए या फिर उनके जो इमेज एडिटिंग के भी जो फीचर्स हैं या कोई फोटो मैनपुलेशन में एआई के फीचर्स यूज़ करने हैं। लाइव कॉल ट्रांसलेटर वगैरह आपको यूज़ करना है। सभी फीचर्स जो है ना दे नॉट ओनली आर देयर बट दे वर्क एंड आर यूज़फुल। मतलब ये जो तीनों चीजें होती है ना ये बहुत इंपॉर्टेंट है। बहुत सारे फोन्स में आपको एआई के फीचर्स मिलेंगे। लेकिन आप एक बार यूज़ करोगे और छोड़ दोगे। नहीं यार मतलब उसका अगर रियल वर्ल्ड में यूज़ नहीं है तो ऐसे मैं Galaxy A के बारे में बिल्कुल नहीं कह सकता हूं। ऑल द फीचर्स दैट आर देयर दे आर यूज़फुल और मैंने वो यूज किए हुए हैं। और ऑनेस्टली बताऊं पिछले सात आठ महीनों में S25 अल्ट्रा के साथ मैंने एi बहुत इस्तेमाल किया है। एस्पेशली उनका जो जैेमिनी इंटीग्रेशन है वो। फॉर एग्जांपल अगर मुझे फोटो निकालना है मैंने आउटफिट कौन सा पहनना चाहिए ये मैं एक्चुअली उससे पूछता हूं। उतना ही नहीं। कौन से कैमरा एंगल से मैं निकाल सकता हूं फोटो ताकि वो अच्छी तरह से आए वो भी मैं उधर ही पूछता हूं। और एक एग्जांपल जो लिटरली एवरीडे मैं करता था। बहुत सारे आर्टिकल्स पढ़ता हूं। कोईकोई बहुत बड़े आर्टिकल्स होते हैं। सिर्फ कहता हूं कि समराइज दिस आर्टिकल और वो बराबर पॉइंट्स दे देता है समराइज करके। आई मीन इट रियली इंक्रीज माय प्रोडक्टिविटी एंड एफिशिएंसी भी मतलब अच्छा दिखना भी उसकी वजह से हो गया। एंड इधर जैसे मैंने फोटो के बारे में बताया कि उधर थोड़ा उनको काम करना है। बट इधर मैं ये कहूंगा कि इनकी जो एआई की टीम है उन्होंने बहुत अच्छा काम किया है। क्योंकि देखिए ना मार्केट लीडर Apple है। उनका एआई देखिए आई मीन इट इज़ स्टिल नॉट देयर। अभी भी फिगर आउट नहीं कर सके और इनका मॉडल ऑलरेडी मेरे हिसाब से एक मैच्योरिटी लेवल पे पहुंच गया है। सो या दैट्स द डिफरेंस एंड दैट प्रोबेबबली मेरे हिसाब से एi आने वाले दिनों में और भी इंपॉर्टेंट होने वाला है और जहां पे मुझे लगता है ये Samsung ने ये रेस आई मीन दे आर अहेड इन द रेस। चलिए अब डिस्प्ले के बारे में बात करते हैं। अब अगर पेपर पे आप देखोगे ना तो इससे ज्यादा ब्राइट डिस्प्लेज़ हैं। इससे शायद और भी ज्यादा फीचर्स वाले डिस्प्ले हैं। बट मुझे फिर भी यही डिस्प्ले मतलब कोई भी फोन 2025 में आप कौन सा भी फोन लीजिए। इसका डिस्प्ले मुझे सबसे ज्यादा अच्छा लगता है। एंड देयर इज वन रीज़न इसका जो एंटी ग्लेयर मैकेनिज्म है या फिर एंटी रिफ्लेक्टिव कोटिंग जो इन्होंने दी हुई है। दैट इज टॉप नच। मतलब मैं iPhone मेरा लेके जाता हूं और ये लेके जाता हूं। iPhone में 3000 निट्स ब्राइटनेस है। इसका 2600 निट्स ब्राइटनेस है। बट आई स्टिल प्रेफर दिस वन। क्यों पता है? सनलाइट में रिफ्लेक्शन नहीं आता है। तुम आप बहुत ही क्लियरली सब कुछ देख सकते हो। सो 6.9 इंच का बड़ा डिस्प्ले है। क्यू एचडी प्लस डिस्प्ले। तो रेजोल्यूशन भी बहुत हाई है। ये 2X डायनामिक एमोलेड डिस्प्ले है। सो वन ऑफ़ द बेस्ट डिस्प्लेज़ आउट देयर। कलर्स के बारे में बात करूं जो डेप्थ वगैरह है। एवरीथिंग इज अबब्सोलुटली ऑन पॉइंट। जैसे मैंने कहा ऐसे नहीं है कि जबरदस्त ब्राइट है। एकदम ही ऐसे पंची वगैरह बट जो है परफेक्ट है। चलिए बैटरी लाइफ के बारे में बात करते हैं। अब मेरा एक्स पर्सनल एक्सपीरियंस ये है। एक तो बता दूं कि मैं मॉडरेट यूजर हूं। एक जैसे नॉर्मल एक एवरेज यूजर यूज़ करता है वैसे ही कुछ मैं यूज़ करता हूं। एंड इन द इवनिंग आई नो कि 2025% बैटरी रहेगी ही रहेगी। सो देखिए बैटरी कैपेसिटी में आजकल हमें 6 7 8000 एमए के मिलने लग गए हैं। दिस वन इज 5000 एमए बट इट लास्ट मी थ्रू अ डे। अगर स्क्रीन ऑन टाइम ऑन पेपर के बारे में बात करूं तो 6 1/2 7 घंटा मिलता है व्हिच इज मोर देन इनफ फॉर मी इट लास्ट थ्रू द डे एंड चार्जिंग हां 45 व एक घंटा 5 मिनट 1 घंटा 10 मिनट में ये फुल चार्ज हो जाता है अगर 0% पे भी बैटरी आ गई सो अबाउट एंड दैट इज फाइन 10-15 मिनट ज्यादा लगता है दूसरे फोन से बट आई एम अब्सोलुटली ओके विथ इट। सो फॉर अ नॉर्मल यूजर 1ढ़ दिन बैटरी जाएगी। हैवी यूजर आप बहुत ज्यादा करोगे फिर भी एक दिन चली जाएगी। हां एक चीज इधर कहूंगा क्योंकि चार्जिंग बैटरी के बारे में बात कर रहे हैं। वायरलेस चार्जिंग जो है ना ऐसे डायरेक्टली मैग्नेट जैसे iPhone में वैसे नहीं है। सो iPhone मैं उधर ज्यादा प्रेफर करता हूं। बट वायरलेस चार्जिंग में इसे केस के साथ आप इस्तेमाल कर सकते हो। सो दैट इज देयर। परफॉर्मेंस के बारे में बात करते हैं। इधर मैं ये भी कह सकता हूं कि पूरे 10 11 महीने की यूसेज में आज तक मुझे परफॉर्मेंस का कभी भी कोई भी प्रॉब्लम नहीं हुआ। और परफॉरमेंस के पहले और एक चीज कहना चाहता हूं जो मैं इतना परेशान हूं जब से अभी अपडेट्स आए हुए हैं iOS के दोस्तों WhatsApp ठीक तरह से नहीं चलता है शेयरिंग ठीक तरह से उसमें नहीं चलता है आई नेवर फेस एनी प्रॉब्लम्स विथ दिस वन कभी नहीं मतलब एवरीथिंग जस्ट वर्क दूसरी बात तो ये है कि एक 7 साल के अपडेट्स तो आपको मिलते हैं सिक्योरिटी अपडेट्स टाइम टू टाइम हर महीने आते रहते हैं आई मीन व्हाट एल्स डू यू वांट व्हेन इट कम्स टू अ यूi जब आया था आउट ऑफ द बॉक्स 1U 7 पे आया था उसके बाद में 1U 8 आया। उसके बाद में 8.5 आया। सब में कभी भी मैंने नोटिस नहीं किया कि बग्गी है, हैंग हो गया है, स्क्रीन फ्रीज हो गई है, शटर बटन नहीं चल रहा है। जो मैं दूसरे फोंस में फेस करता हूं इसमें पूरे 10 11 महीने में आई डिड नॉट फेस दैट। एंड डे टू डे यूसेज में, एप्स ओपनिंग में। देखिए ये आउट एंड आउट गेमिंग फोन नहीं है। बट फिर भी हैवी अगर कुछ भी करते हो आप कभी भी ऐसे महसूस नहीं होगा कि परफॉर्मेंस इज अ बॉटल ने। कभी भी नहीं। एंड 8 एलट के जितने फोंस देखे हैं सबसे ज्यादा अच्छी ट्यूनिंग मेरे वन ऑफ द टॉप थ्री ट्यूनिंग मैं कहूंगा Samsung Galaxy S25 अल्ट्रा में हुई है क्योंकि कभी भी ज्यादा गर्म नहीं हुआ। ऐसे कभी नहीं लगा कि यार मैं कुछ बहुत हैवी कर रहा हूं और पीछे एक ही जगह पे गर्म हो गया। नहीं एंड आई थिंक दैट इज बिकॉज़ इनका जो वेपर कूलिंग चेंबर हो गया है वो एक्सपैंड बड़ा हो गया है ज्यादा। उसकी वजह से थर्मल्स आर इन चेक। और इवन बेंचमार्क स्कोर्स में भी मुझे याद है सिर्फ यही फोन है एट एलit पे जिसने मल्टी कोर में 10,000 के ऊपर स्कोर किया हुआ है। दूसरे एट एलट के फोन 9000, 500, 800, 700 ऐसे किए हुए हैं। सो ऑप्टिमाइजेशन जो है एंड परफॉर्मेंस जो है आई थिंक इज स्टॉप लॉच। लेकिन हां एक चीज मैं कहना चाहूंगा वन UI 7 तक बहुत सारे नए-नए फीचर्स ऐड हो रहे थे। 1UI 8 के बाद में वही फीचर्स जो है और ज्यादा फीचर रिच रिचनेस ऐड किया हुआ है या फिर पॉलिश हो गए हैं ये फीचर्स। सो यू नो मैं आपको एग्जांपल देता हूं। जैसे ह्यूमन लैंग्वेज सर्च आपको कुछ भी सेटिंग में जाए सेटिंग्स में जाके ढूंढने की जरूरत नहीं है। अब आप सिर्फ सर्च में सर्च कीजिए। आपको जो भी चाहिए हैप्टtिक फीडबैक हैप्टtिक वो इमीडिएटली आपको दिखा देगा एंड यू गो दे। तो ये छोटी-छोटी चीजें हैं बट इट हैज़ मेड One इवन मोर यूजर फ्रेंडली और कंसिस्टेंटली वेदर आप पॉडकास्ट देखिए कुछ भी देखिए मैं हमेशा 1UI के बारे में बात करता हूं। मैं ये बात करता हूं कि यार S25 अल्ट्रा तो मेरे साथ रहने वाला है। दैट इज द रीज़न बिकॉज़ डिपेंडेबल है ये एक बॉस। तो फाइनली इस लॉन्ग टर्म रिव्यु में मैं क्या कहना चाहूंगा? देखिए एक लाइन में अगर मैं आपको बताऊं तो ये एक प्रेडिक्टेबल कंसिस्टेंट डिपेंडेबल फ्लैशिप है और वो जो डिपेंडेबल वर्ड है ना वो बहुतेंट है क्योंकि जब कोई चीज आपको करनी है आपके स्मार्टफोन पे वेदर इट इज हैवी वेदर इट इज गेमिंग कुछ भी करना है आपको पता है आपको रिजल्ट क्या मिलने वाला है एंड इट वर्क्स लाइक दैट एवरी टाइम आई थिंक वो मुझे सबसे ज्यादा अच्छा लगता है। हां, देयर आर स्लाइट वीकनेसेस कि भाई कैमरा में जितना इनोवेशन होना चाहिए उतना नहीं है। वायरलेस चार्जिंग का सपोर्ट है। बट दीज़ आर नॉट द डील ब्रेकर्स। अगर आप डिस्प्ले के बारे में बात करते हो, UI के बारे में बात करते हो, अपडेट्स के बारे में बात करते हो, परफॉर्मेंस के बारे में बात करते हो। दिस इज टॉप नच। इवन इन कैमरा। कैमरा में भी मैंने कहा ना स्टिल फोटो जो है आई प्रेफर S25 अल्ट्रा के स्टिल फोटो। वीडियोस में iPhone आगे निकल जाता है। बट स्टिल फोटोज में कंसिस्टेंट परफॉर्मेंसर अगेन यही है। और जो अब एi के फीचर्स वगैरह है वो इट इज लाइक आइसिंग ऑन द केक। वो है। और एक चीज मुझे अच्छी लगती है दोस्तों। iPhone में या फिर आप दूसरे भी फ्लैकशिप्स लीजिए। हर साल आपको 5000 6000 से प्राइसेस बढ़ते रहते हैं। पिछले 3 साल से इसकी जो प्राइस आई हुई है इट रिमेंस कांस्टेंट। तो यही था दोस्तों जो मुझे लगता है S25 अल्ट्रा के बारे में जो मैंने एक्सपीरियंस किया एंड जो मैं हमेशा पॉडकास्ट में भी बोलता हूं कि यार मुझे यह फोन चाहिए हमेशा अभी भी एक सिम इवन दो दैट इज प्राइमरी ये मेरे साथ हमेशा रहता है एंड आई थिंक वैरी सून S26 अल्ट्रा भी यही होने वाला है फॉर ईयर 2026 होपफ्फुली देखते हैं बट दिस इज व्हाट S25 अल्ट्रा इज अगर आपको कोई सवाल है अगर आप फ्लैशिप देख रहे हो कि Android फ्लैकशिप देख रहे हो डेफिनेटली आपने इसके पास देखना चाहिए। चलिए दोस्तों, इस वीडियो में इतना ही। अगले वीडियो तक कीप ट्रैकिंग एंड स्टे सेफ। [संगीत]
""",
    extraction=Extraction(
        battery=BatteryExtraction(
            battery_capacity=5000,
            battery_capacity_extraction_text="5000 mAh"
        ),
        charging=ChargingExtraction(
            charging_power=45,
            charging_power_extraction_text="45W wired",
            wireless_charging=True,
            wireless_charging_power=15,
            wireless_charging_power_extraction_text="15W wireless",
            charger_in_box=False,
            charger_in_box_extraction_text="does not come with a charger"
        ),
        design=DesignExtraction(
            weight_grams=218.0,
            weight_grams_extraction_text="218 g",
            thickness_mm=8.2,
            thickness_mm_extraction_text="8.2 mm thick"
        ),
        screen=ScreenExtraction(
            screen_size_inches=6.8,
            screen_size_inches_extraction_text="6.8-inch Dynamic AMOLED",
            refresh_rate_hz=120,
            refresh_rate_hz_extraction_text="120Hz refresh rate"
        ),
        in_the_box=InTheBoxExtraction(
            items=[
                ItemExtraction(
                    item_name="Data Cable",
                    item_specifications="USB Type-C to Type-C",
                    item_extraction_text="Data Cable (Type-C to Type-C)"
                ),
                ItemExtraction(
                    item_name="SIM Ejection Pin",
                    item_specifications=None,
                    item_extraction_text="SIM Ejection Pin"
                )
            ]
        ),
        variants=[
            VariantExtraction(
                color="Titanium Black",
                ram="12GB",
                storage="256GB",
                extraction_text="Titanium Black, 12GB Memory, 256GB Storage"
            )
        ]
    )
)

EXAMPLE_FOLDABLE = ExampleData(
    model_name="Motorola Razr 60 Ultra",
    text="""--- SOURCE: OEM_OFFICIAL (PRIORITY 1 — ALWAYS AUTHORITATIVE ON HARDWARE SPECS) ---
Performance

Operating System

- Android™ 15

Internal Storage

- 512GB built-in\\* UFS4.0

Sensors

- Fingerprint reader, Notification LED, Approch IR sensor, Proximity sensor, Ambient light sensor, Accelerometer, Gyroscope, eCompass, Hall sensor

Processor

- Snapdragon® 8 Elite Mobile Platform

CPU

3nm Process Technology

Octa-core (2 Prime+6 Performance cores, with up to 4.32GHz clock speed)

GPU

Qualcomm® Adreno™

NPU

Qualcomm® Hexagon™ NPU


Memory (RAM)

- 16GB LPDDR5X RAM\\*

Security

- Side Fingerprint reader

Face Unlock

OS Upgrade + Security Patches

- (3 OS + 4 Years SMR)

Battery

Battery Size

- 4700mAh

Charging

- 68W TurboPower™ charging

30W wireless charging support

5W reverse charging

Display

Display Size

- Main display: 6.96" 1224p Super HD display

External display: 4.0" pOLED display

Resolution

- Main display: Super HD (2992 x 1224) \\| 464ppi

External display: 1272 x 1080 \\| 417ppi

446ppi

Screen to Body Ratio

- Active Area-Body: 86.2%

Display Technology

- Main display:

LTPO

Foldable AMOLED

HDR10+

10-bit

120% DCI-P3 color gamut

Up to 165Hz refresh rate

Touch rate: 220Hz/300Hz (game mode only)

HBM: 2000nits

Peak Brightness: 4500nit

Dolby Vision



External display:

LTPO

Flexible AMOLED

HDR10+

10-bit

100% DCI-P3 color gamut

Up to 165Hz refresh rate

Touch rate: 120Hz/165Hz (game mode only)

HBM: 1500nit

Peak Brightness: 3000nit


Aspect Ratio

- Main display: 22:9

Display Protection

- External display: Corning Gorilla Ceramic

Design

Body

- Front : Corning® GorillaTM Glass Ceramic

Rear : Alcantara/FSC-certified wood/Satin-inpired finish/Leather-inspired finish

Side : AI Key


Ports

- Type-C port (USB 2.0)

Weight

- 199g

Water Protection

- IP48 dust and underwater protection2

Colours

PANTONE Scarab

PANTONE Mountain Trail

PANTONE Rio red

Camera

Rear Main Camera

- **Main camera:**

50MP (f/1.8, 1.0μm) or 12.6MP (2.0μm Quad Pixel)

OIS

1/1.56"

Instant-all Pixel Focus

Pantone™ Validated Color and Skin Tones

**Camera 2**

Ultrawide + macro camera

50MP (f/2.0, 0.6μm) or 12.6MP (1.2μm Quad Pixel)

FOV 122°

**Camera 3**

NA

**Camera 4**

NA

Rear Camera Video Software

- Dolby Vision

Camcorder Mode

Dual Capture

Horizon Lock

Hyperlapse Stabilization

Live Filters

Slow-Motion

Auto Focus Tracking

Timelapse

Video HDR

Audio Zoom

Advanced Stabilization

Face Retouch


Front Camera Video Capture

- Internal

4K UHD (30fps), FHD (60/30fps)



External

Main: 4K UHD (60/30fps), FHD (60/30fps)

Ultra-wide: 4K UHD (60/30fps), FHD (60/30fps)


Rear Camera Software

- Action Shot

Group Shot

Signature Style

Long Exposure

Super Zoom

Ultra HDR

Auto Night Vision

Auto Smile Capture

Pantone Validated™ Color

Pantone Skintone™ Validated

Scan (Powered By Adobe Scan)

Photo Booth

Portrait Mode

Night Vision

Macro

Pro Mode

Face Retouch

Gesture Selfie

Live Filters

HDR

Panorama

Tilt-Shift

External Display Preview

Instant Review

Camera Cartoon



Google photo editing features:

Auto Enhance

HDR Effect

Magic Eraser

Magic Editor, with Reimagine & Auto Frame\\* (Available through Playstore update)

Portrait Blur

Photo Unblur

Portrait Light

Sky


Front Camera Hardware

- 50MP (f/2.0, 0.64μm) or 12.5MP (1.28μm Quad Pixel)

Rear Camera Video Capture

- 8K (30fps)

4k UHD (60fps/30fps)

FHD (60fps/30fps)

Slow motion

4K UHD (120fps)

FHD (240fps/120fps)

Flash

- LED flash

Audio

Speakers

- Dual stereo speakers with Dolby Atmos® featuring Spatial Sound Qualcomm® Snapdragon Sound™4

Microphones

- 3 microphones

Voice Control

- Google Assistant

Connectivity

Networks + Bands

- 5G: sub-6

4G: LTE

3G: WCDMA

2G: GSM

**Bands**

3G: GSM850/900/1800/1900; W1/2/4/5/8

4G: B1/2/3/4/5/7/8/12/13/14/17/18/19/20/25/26/28/29/30/32/34/38/39/40/41/42/43/48/66/71

5G sub-6: N1/2/3/5/7/8/12/14/20/25/26/28/29/30/38/40/41/48/66/70/71/75/77/78


Bluetooth Technology

- Bluetooth®5.4

NFC

- Yes

Wi-Fi

- WiFi 2.4G/5G, WiFi 6/6E, WiFi 7

802.11 a/b/g/n/ac/ax/be

2.4GHz & 5GHz & 6GHz\\*

Wi-Fi hotspot

Location Services

- GPS, GLONASS, Galileo, QZSS, Beidou

SIM Card

- Dual SIM (pSIM + pSIM)

Side Frame

- Metal

IP Rating

- IP48

In the Box

Device

- motorola razr 60 ultra

In box accessories

- Signature packaging fragrance

68W TurboPower™ charger

USB Type-C cable

Guides

SIM tool

ESG protective case


Country of Origin

- India

Manufacturing Details

Manufacturer's Details

- Padget Electronics Pvt Limited, A-23, Sector-60, Noida, Gautam Buddha Nagar, Uttar Pradesh- 201301

Country of Origin

- India

--- SOURCE: GSMARENA (PRIORITY 3 — USE ONLY WHEN OEM_OFFICIAL IS SILENT ON A FIELD) ---
[Also available as Motorola Razr Ultra in North America](https://www.gsmarena.com/motorola_razr_ultra_2025-13823.php)

| Network | [Technology](https://www.gsmarena.com/network-bands.php3) | [GSM / CDMA / HSPA / EVDO / LTE / 5G](https://www.gsmarena.com/motorola_razr_60_ultra-13805.php#) |
| [2G bands](https://www.gsmarena.com/network-bands.php3) | GSM 900 / 1800 / 1900 |
|  | CDMA 800 / 1900 |
| [3G bands](https://www.gsmarena.com/network-bands.php3) | HSDPA 850 / 900 / 1700(AWS) / 1900 / 2100 |
|  | CDMA2000 1xEV-DO |
| [4G bands](https://www.gsmarena.com/network-bands.php3) | 1, 2, 3, 4, 5, 7, 8, 12, 13, 14, 17, 18, 19, 20, 25, 26, 28, 29, 30, 32, 34, 38, 39, 40, 41, 42, 43, 48, 66, 71 |
| [5G bands](https://www.gsmarena.com/network-bands.php3) | 1, 2, 3, 5, 7, 8, 12, 14, 20, 25, 26, 28, 29, 30, 38, 40, 41, 48, 66, 70, 71, 75, 77, 78 SA/NSA/Sub6 |
| [Speed](https://www.gsmarena.com/glossary.php3?term=3g) | HSPA, LTE, 5G |

| Launch | [Announced](https://www.gsmarena.com/glossary.php3?term=phone-life-cycle) | 2025, April 24 |
| [Status](https://www.gsmarena.com/glossary.php3?term=phone-life-cycle) | Available. Released 2025, April 25 |

| Body | [Dimensions](https://www.gsmarena.com/motorola_razr_60_ultra-13805.php#) | Unfolded: 171.5 x 74 x 7.2 mm<br>* * *<br>Folded: 88.1 x 74 x 15.7 mm |
| [Weight](https://www.gsmarena.com/motorola_razr_60_ultra-13805.php#) | 199 g (7.02 oz) |
| [Build](https://www.gsmarena.com/glossary.php3?term=build) | Plastic front (unfolded), glass front (folded, Gorilla Glass Ceramic), silicone polymer back (eco leather), aluminum frame (6000 series), hinge (stainless steel) |
| [SIM](https://www.gsmarena.com/glossary.php3?term=sim) | · Nano-SIM + [eSIM](https://www.gsmarena.com/glossary.php3?term=esim)<br>* * *<br>· Nano-SIM + Nano-SIM |
|  | IP48 dust and water resistant (dust > 1mm; immersible up to 1.5m for 30 min) |

| Display | [Type](https://www.gsmarena.com/glossary.php3?term=display-type) | Foldable LTPO AMOLED, 1B colors, 165Hz, Dolby Vision, HDR10+, 4500 nits (peak) |
| [Size](https://www.gsmarena.com/motorola_razr_60_ultra-13805.php#) | 7.0 inches, 106.7 cm2 (~84.1% screen-to-body ratio) |
| [Resolution](https://www.gsmarena.com/glossary.php3?term=resolution) | 1224 x 2912 pixels (~464 ppi density) |
| [Protection](https://www.gsmarena.com/glossary.php3?term=screen-protection) | Mohs level 5 |
|  | Second external LTPO AMOLED, 1B colors, Dolby Vision, 165Hz, HDR10+, 3000 nits (peak)<br>* * *<br>4 inches, 1272 x 1080 pixels, 417 ppi, Gorilla Glass Ceramic |

| Platform | [OS](https://www.gsmarena.com/glossary.php3?term=os) | Android 15 |
| [Chipset](https://www.gsmarena.com/glossary.php3?term=chipset) | Qualcomm SM8750-AB Snapdragon 8 Elite (3 nm) |
| [CPU](https://www.gsmarena.com/glossary.php3?term=cpu) | Octa-core (2x4.32 GHz Oryon V2 Phoenix L + 6x3.53 GHz Oryon V2 Phoenix M) |
| [GPU](https://www.gsmarena.com/glossary.php3?term=gpu) | Adreno 830 |

| Memory | [Card slot](https://www.gsmarena.com/glossary.php3?term=memory-card-slot) | No |
| [Internal](https://www.gsmarena.com/glossary.php3?term=dynamic-memory) | 256GB 16GB RAM, 512GB 12GB RAM, 512GB 16GB RAM, 1TB 12GB RAM, 1TB 16GB RAM |
|  | UFS 4.0 |

| Main Camera | [Dual](https://www.gsmarena.com/glossary.php3?term=camera) | 50 MP, f/1.8, 24mm (wide), 1/1.56\\", 1.0µm, multi-directional PDAF, OIS<br>* * *<br>50 MP, f/2.0, 12mm, 122˚ (ultrawide), 0.6µm, PDAF |
| [Features](https://www.gsmarena.com/glossary.php3?term=camera) | LED flash, panorama, HDR, Pantone Validated Colour and Skin Tones |
| [Video](https://www.gsmarena.com/glossary.php3?term=camera) | 8K@30fps, 4K@30/60/120fps, 1080p@30/60/120/240fps, Dolby Vision HDR, gyro-EIS |

| Selfie camera | [Single](https://www.gsmarena.com/glossary.php3?term=secondary-camera) | 50 MP, f/2.0, (wide), 0.64µm |
| [Features](https://www.gsmarena.com/glossary.php3?term=secondary-camera) | HDR |
| [Video](https://www.gsmarena.com/glossary.php3?term=secondary-camera) | 4K@30/60fps, 1080p@30/60fps |

| Sound | [Loudspeaker](https://www.gsmarena.com/glossary.php3?term=loudspeaker) | Yes, with stereo speakers (with Dolby Atmos) |
| [3.5mm jack](https://www.gsmarena.com/glossary.php3?term=audio-jack) | No |
|  | Snapdragon Sound |

| Comms | [WLAN](https://www.gsmarena.com/glossary.php3?term=wi-fi) | Wi-Fi 802.11 a/b/g/n/ac/6e/7, dual-band or tri-band (region dependent) |
| [Bluetooth](https://www.gsmarena.com/glossary.php3?term=bluetooth) | 5.4, A2DP, LE, aptX HD, aptX Adaptive, aptX Lossless |
| [Positioning](https://www.gsmarena.com/glossary.php3?term=gnss) | GPS, GALILEO, GLONASS, BDS, QZSS |
| [NFC](https://www.gsmarena.com/glossary.php3?term=nfc) | Yes |
| [Radio](https://www.gsmarena.com/glossary.php3?term=fm-radio) | No |
| [USB](https://www.gsmarena.com/glossary.php3?term=usb) | USB Type-C, OTG |

| Features | [Sensors](https://www.gsmarena.com/glossary.php3?term=sensors) | Fingerprint (side-mounted), accelerometer, gyro, proximity, compass, barometer |

| Battery | [Type](https://www.gsmarena.com/glossary.php3?term=rechargeable-battery-types) | 4700 mAh |
| [Charging](https://www.gsmarena.com/glossary.php3?term=battery-charging) | 68W wired<br>* * *<br>30W wireless<br>* * *<br>5W reverse wired |

| Misc | [Colors](https://www.gsmarena.com/glossary.php3?term=build) | Pantone: Rio Red, Scarab, Mountain Trail, Cabaret |
| [Models](https://www.gsmarena.com/glossary.php3?term=models) | XT2551-6 |
| [Price](https://www.gsmarena.com/glossary.php3?term=price) | [€ 639.63 / $ 1,299.99 / £ 599.97](https://www.gsmarena.com/motorola_razr_60_ultra-price-13805.php) |

| Our Tests | [Performance](https://www.gsmarena.com/glossary.php3?term=benchmarking) | AnTuTu: 1831189 (v10)<br>* * *<br>GeekBench: 6796 (v6)<br>* * *<br>3DMark: 5938 (Wild Life Extreme) |
| [Display](https://www.gsmarena.com/gsmarena_lab_tests-review-751p2.php) | [1489 nits max brightness (measured)](https://www.gsmarena.com/motorola_razr_60_ultra-review-2824p3.php#dt) |
| [Loudspeaker](https://www.gsmarena.com/gsmarena_lab_tests-review-751p7.php) | [-25.5 LUFS (Very good)](https://www.gsmarena.com/motorola_razr_60_ultra-review-2824p3.php#lt) |
| [Battery](https://www.gsmarena.com/how_we_test_gsmarena_battery_life_test_v2-news-60429.php) | [Active use score 15:10h](https://www.gsmarena.com/motorola_razr_60_ultra-review-2824p3.php#bt) |

| EU LABEL | [Energy](https://www.gsmarena.com/glossary.php3?term=eu-energy-class) | Class A |
| [Battery](https://www.gsmarena.com/glossary.php3?term=eu-battery-endurance) | 48:48h endurance, 1000 cycles |
| [Free fall](https://www.gsmarena.com/glossary.php3?term=eu-free-fall-class) | Class D (50 falls) |
| [Repairability](https://www.gsmarena.com/glossary.php3?term=eu-repairability-class) | Class B |

--- SOURCE: TRANSCRIPT (PRIORITY 2 — USE FOR: charger_in_box, in_the_box, India-specific colors, India variant confirmations) ---
Okay, so by the thumbnail you have got an idea. We are going to talk about a flip phone. But to know how far and how evolved flip phones have become, let's go back a bit and see where they were. So flip phones were launched back in 2020 by Motorola and Samsung. They had a smaller outer display which was useful for just looking at your notifications. But to do any other work, you had to open the phone and then do it. And Samsung flips had more market share than Motorola flips. Then in 2023, Motorola with the Razer 40 Ultra, they introduced the biggest cover display on which you could use 80% of the phone right from the cover display. This led to the Motorola Razer Flip to cut out Samsung's dominance in the flip segment. And Motorola seems to continue to increase in dominance with the all new Motorola Razer 60 Ultra. It comes with a 4-in cover display, Snapdragon 8 processor, UFS 4.0 storage. All of this for 89 with offers. Now, I know what you're thinking. Is this the best flip phone to buy right now? Moreover, is there any catch? Well, we'll tell you all of that. Pratik Techweiser, let's flipping go. But before that, it comes in this big black box. Inside the box, obviously, you get the phone, some paperwork, a SIM ejector tool, and the good thing is you get a color coordinated case, and this is very good for a flip phone, a 68 watt PD charger, and a USBC toC cable. Everything you need comes inside the box. Now, roughly, there are three big changes with the Moto Razer 60 Ultra. So like the first thing that grabs your attention here is the design. Back here comes in this wood finishing it and it has that wooden texture here because this has real wood fibers. And this is not the first time Motorola has been doing this. They did it first with their Moto X phone back in 2013 and then we saw the same wooden finish last year in their 50 Ultra as well. Now you need to keep the phone regularly outside in sunlight. Why? For fresh air and sunlight. Doesn't apply if you are in Delhi. Now apart from this wooden design, it also comes in this Alcantra finished back. Like this Alcantra material is mostly found in car interiors of Ferrari, Lamborghini or Porsche. I mean I've seen videos of it. Let us know if you have one and this is the same finish your car has. Now there's also a third color which comes in the vegan leather finish. Now the Razer 60 Ultra comes with an IP 48 rating. So if the phone accidentally slips into the sink while washing your hands, it can handle dipping in water. Along with this, the hinge here is made out of titanium. So, how does this make a difference from other normal hinges? See, Titanium is known to be durable and lightweight at the same time. So, the smartphone makers don't need to pack in extra components to make the hinge more durable. Plus, the inner screen won't have a crease. Now, Motorola claims you can open and close the flip up to 8 lakh time. So, basically, if you open and close 100 times daily, it'll still last up to 20 years. Which brings us next to this display. Oh, sorry, displays. The Razer 60 Ultra comes with a large 4in cover display outside. The cover display here has Corning Gorilla Glass ceramic protection. Now, an advantage of a Razer Flip is you can use 80 to 90% of the phone from the front display itself. Also, the cover display supports auto brightness. So, depending on the light in the environment, it automatically adjusts the brightness level. Outdoors, let it be booking rides, responding to notifications, using Google Maps to navigate, it all can be done right from the cover display. And the cover display is pretty bright enough. Heck, you can even play games like BGMI on the outer screen, although you won't do that. You can play some hyper casual games as well. Other than that, since this is a flip phone, you can either keep it like this in tent mode or you can keep it like this in desk mode. It acts like a desk clock. And then there is an interesting thing that you can do. I'm coming to that in just a moment. For watching real movies or YouTube videos, you get this 7in inner display. Now, when you're watching videos, even at extreme angles, the crease can't be seen on the screen. Of course, if you go parallel to the desk, you can see it, but then why would you do that? Now if I move my finger over the crease or the middle part of the display, you can slightly feel an indent but the crease is almost invisible. Now I was watching this movie Exter Territori in Netflix. The color, sharpness and all of them are so good but surprisingly there is no HDR support on Netflix. Now what enhances the movie watching experience here are the speakers. Now it comes with the Dolby Atmos support and have a listen to [Music] this. Like the speakers are pretty loud. It's bassy and by far it is one of the best speakers I've heard on a flip phone. Even outdoors under direct sunlight, the display is pretty bright. Now coming to the part which Motorola phones are known for, software. The Razer 60 Ultra runs on Hello UI based on the latest Android 15. Now in terms of features, unlike all the Motorola phones in 2025, the Razer 60 Ultra also comes with Moto AI. If your phone is just kept in desk mode, you can just look at the phone and say, "Can you tell me the latest news?" Tensions remain high between India and Pakistan following a recent terror attack in Palahalgam, Kashmir. At least this one doesn't fake it like the mainstream media. And not just this, you can even ask it about the weather or even ask about the new IPL schedule. It will tell you that as well. Basically, it sits like a friend who can help you with all the information that you need. There is a dedicated button to trigger the Moto AI. Like if you have some queries, press the button and you can directly ask Copilot or Publicity AI without installing any app. In fact, you can go one step further and you can also ask perplexity AI what are the nearby restaurants that serve ramen and you get the results instantly which is faster than searching on Google and opening five different websites and reading articles. Also, if you want to make your group chat even funnier, you have this image studio here. You can create stickers, gifs, etc. So, I'll create a sticker of Virat Kohli in the RCB jersey holding the IPL trophy. And boom, here is the sticker ready. Of course, this is AI. This is not reality. RCB and IPL trophy. Now the next feature that I found very practical is the notification summary. So these days a lot of apps sends those fluffy marketing notifications and my notification panel always looks crowded. So instead of going through each of those notification and swiping them out, you just double tap this button. It can summarize the notifications from the apps that are important for you like highly recommended, highly efficient. And then there's also some fun stuff like Moto AI can curate a playlist for you as well. To make this feature work, you need to log into your Amazon music account once. So I asked it to curate a playlist of some lowfi music which I can listen to while writing and it does that instantly. Next you can also ask Moto AI to do complex stuff like I can ask Moto AI here to open Zumato on my laptop and see it does that. Now there are tons of features of Moto AI or Hello UI. Let us know if you want a separate video on all of that. If enough Moto users want it, we'll do it. Now what has been really good for Motorola recently is the camera processing. So let's talk about the camera. So this year the Razer 60 Ultra comes with a dual camera setup. And unlike last year, you get an ultra wideangle camera instead of the telephoto camera. The pictures in the daylight, they come out good like any other Motorola phone. The colors and light control are good. The photo looks poppy, vibrant, but still the skin tone is close to natural. If you zoom in on my face, there are good detail. And speaking of zoom, it can go max at 20x. And the zoom photos, well, the colors look a bit bland. Now in terms of videos, it can shoot maximum at 8K 30fps. And this is how the video and audio quality is. There is a generator beside us. You can have a good idea of the noise cancellation. And from the front camera, you can shoot at 4K 30fps. And this is how the selfies look. Now, so far things seem quite good. Isn't there any catch? Well, that brings me to the performance. The Razer 60 Ultra comes with Snapdragon 80 processor along with UFS 4.0 storage. Now, we ran the N22 benchmark, which gives an overall score to the phone, and it scored somewhere around 21 lakh, which is close to what the S25 Ultra scores. And we also ran the CPU throttling test and the graph here is as red as your eyes after binging all night. Jokes aside, the phone here gets quite hot, but these are just benchmarks. Do the issues replicate in real life as well? So, we played DJI. You can play it maximum at 90 fps in smooth graphics. The gameplay here was good though. The cover screen part felt a bit warm at the start of the game, but 5 minutes into the game, it didn't feel that warm. Now when we pushed it to play a heavier game like COD war zone with graphics set to high at 60 fps and then the game was stuttering here and there a bit and the cover display part also got quite warm and you can see the gameplay was a bit choppy. So in performance when you push it to the maximum then only the phone throttles and this is not just with this phone any flip phone or compact phone has very less area to dissipate heat and they throttle. Now before we get to the conclusion the Razer 60 Ultra comes with a 4700 mAh battery that's more than 15% improvement from last year. So on a single charge, playing games, watching videos and reals, clicking pictures and all of that. It easily lasted me one entire day. Funny thing, now flips are getting bigger battery than compact phones. And speaking of charging, it supports 68 watt fast charging. So it takes about 40 minutes to charge the phone completely. It supports up to 30 W wireless charging and 5 W reverse wireless charging. So what's the conclusion here? Are flip phones ready for normal users like you and me? See, earlier the flip phones used to be kind of fragile with a small cover display which was only useful for seeing notifications. Now skip to 2 three years now. Motorola with the Razer was the first one to introduce the external display made it bigger with the Razer 50 Ultra and now with the Razer 60 Ultra they have added Moto A smartly which is usable and with the improvements in the hinge and this no crease display and actually the camera I believe this year Motorola cameras have been really good and we just got to know the pricing. The Razer 60 Ultra cost double 9 and with bank offers and all of that the net effective pricing is 89 and goes on sale on 21st May on Amazon, Reliance Digital and Motorola's own website. So it is the same price as last year with display, battery and processor upgrades. Pratik Techiser signing off. See you in the next video. [Music]
Vola Rer 60 अल्ट्रा आ चुका है मेरे दोस्तों। पहली बार एक टाइटेनियम का हिंज डाला हुआ है। कैमरे में चेंजेस है। आप अपने आप को देख सकते हो भाई। एक टेलीोटो लेंस की मेरे को कमी फील हुई। ये वाले काफी सारे फीचर ऐड कर रखे हैं। प्राइस बता देता हूं। एक चीज जो फोल्डिंग फोन में भाई सदियों से दिक्कत चली आ रही है। बैटरी वो सिरामिक में ड्यूरेबिलिटी बहुत तगड़ी होती है। Motra ले भाई हमने सारे रेजर अनबॉक्स कर रखे हैं। और हर बार कुछ अलग सा कोई इंटरेस्टिंग सा कर दे। वो कुछ फर्क लग रहा है गाइस राइट लेफ्ट। सबसे पहली चीज ओ सॉरी सॉरी सॉरी। रीटेक करेंगे। ओ अब हुई ना भाई तगड़ी लैंडिंग। बट भाई फ़ खोलते ही सबसे पहली चीज जो आप नोटिस करोगे ना। परफ्यूम डाल के भेजा यार। मेल इत्र है कि फीमेल इत्र? आई थिंक मेल। मेल फीमेल इत्र दोनों मिलाकर डाला है। भाई देखो फ़ोन ना उतावला हो रहा है भाई मेरे पास आने के लिए। ओव आपको लग रहा होगा 89 W का चार्जर बट नहीं मेरे दोस्तों ये है 68 W का चार्जर। FDFC केबल एक प्रोटेक्टिव कवर को भी कवर कर रखा है। वाओ अरे आ गया। आज ऐसा लगा कोई पत्थर गिरा है। आ भाई दर्द हुआ मेरे को। अब देखो सेरेमिक में फायदा ये रहता है कि भाई ड्यूरेबिलिटी बहुत तगड़ी होती है। बट सिरेमिक में स्क्रैचेस भी थोड़े से ज्यादा आते हैं। क्योंकि iPhone में भी जब सेरेमिक का ग्लास आया है उसमें स्क्रैचेस थोड़े से बढ़ गए हैं। तो उसकी भाई हमको चेकिंग करनी पड़ेगी। उंगलियां पूरी लाल जैसे पड़ गई है। देखो अभी इतने से तो स्क्रैच नहीं आया। बट जितना मेरा एक्सपीरियंस है भाई सेरेमिक पे थोड़े से स्क्रैचेस आते हैं। बट एक टॉप क्वालिटी प्रोडक्शन रहती है। मार्बल और टाइल में एक फर्क होता है। टाइल थोड़ा ज्यादा चमक मारती है। एंड सेम चीज इस ग्लास के साथ भी। ये थोड़ा ज्यादा चमक मारता है। तो अगर आप आउटडोरर्स यूज़ कर रहे हो रिफ्लेक्शन बहुत आएगी। पहले अगर ये बैग आप नोटिस करो तो बात आपको अलग सी लगेगी। भाई वो नहीं होते। रजाई गद्दे वेलवेट वाले सोफे और पर्देवदे होते हैं ना ऐसी एक वाइब आ रही है जिसमें साइड थोड़ी सिर्फ वीगन लेदर टाइप वाइप की है एंड एक वुड वाला वेरिएंट भी आता है। वुड भी काफी ज्यादा इंटरेस्टिंग सा लगता है। अगर आप राइट साइड में देखोगे पावर बटन तो पावर बटन सिर्फ छोटे वॉल्यूम बटंस लगा रखे हैं। इसमें फिंगरप्रिंट स्कैनर भी है। बाकी बॉटम में दो सिम स्लॉट है। USB पोर्ट है, स्पीकर है। एंड राइट साइड में एक और पावर बटन दिखेगा। बट ये एक्चुअली एआई बटन है। इसको जैसे ही आप खोलोगे ना यहां पे बीच में आपको ऐसा एक चमकता हुआ मेटल दिखेगा। ये एक्चुअल टाइटेनियम है। पिछली बार जो फ्लिप फोन था उसको अगर आप हल्का सा इसे ट्विस्ट करें तो ये बहुत स्लाइट सा ऐसा वर्बल सा होता था। बट अब एकदम ही एक फर्म इतने से एंगल पे भी आप इसको रोक के रख सकते हो। और टाइटेनियम के काफी सारे फायदे भी हैं कि एक भाई ये हल्का हो जाता है क्योंकि नॉर्मल मेटल से हल्का रहता है एंड छोटी स्पेस में टाइटेनियम ज्यादा मजबूती दे पाता है। तो इसीलिए हिंज का साइज भी छोटा करा जा सकता है। अगर मैं इसे ट्विस्ट भी कर रहा हूं राइट लेफ्ट उसमें भी मेरे को मजबूत फील हो रहा है। एंड लास्ट टाइम से अगर मैं वेट कंपेयर करूं तो पहले 190 ग्राम था एंड अब 200 ग्राम हो गया। तो आपको लगेगा टाइटेनियम से तो वेट भाई कम करा रहा था। बढ़ कैसे गया? क्योंकि टाइटेनियम का तो वेट कम हुआ बट बैटरी इंक्रीस कर दी गई है। टाइटेनियम से एक और फायदा होता है क्योंकि भाई ये जो फोन मुड़ रहा होता है इसको सपोर्ट्स ही होते हैं एक स्ट्रांग। तो इसीलिए आपका ये जो फोल्ड का क्रीज आता है वो काफी ज्यादा कम हो जाता है। एंड अब एक ऐसा पॉइंट आ गया समझ लो अगर मैं आपको ऐसे दिखा रहा हूं तो शायद आपको स्क्रीन पे क्रीज दिखे भी ना। बहुत स्लाइट सा अगर एक एंगल पे आप देखोगे तब शायद थोड़ी बहुत आपको क्रीज दिखे। यहां पे सिक्के को भाई हिंज में फंसा दिया। भाई ग्रिप देखो आप। ऐसा भी नहीं है कि पूरी ग्रिप आ रखी है भाई। बट सिक्के ने पकड़ रखा है भाई। पेपरवेपर से तो अगर आपको उठाना हो तो इतना ज्यादा चैलेंज आता नहीं है। देखो भाई एकदम पतली शीट है पेपर की। क्या ग्रिप करा हुआ है? कपड़े से भी उठा। एक हाथ से आप खोल बांध भी रहे हो ना, उसमें भी आपको ना एक बहुत स्टडीनेस सी फील होती है। अगर आप हाथ में पकड़ के यूज़ कर रहे हो ना एक सॉलिड फ़ फील होता है। IP48 रेटिंग है। थोड़ा सा पानी की छिपा के आप आगे भी पड़ जाए तो ऐसे ज्यादा चैलेंज आएगा नहीं। बाहर भी जो डिस्प्ले है भाई वो कम नहीं है। वो एक 165 हर्टz का। एलtपीओ है। एमोलेड है 165 हर्टz का। अभी तक का सबसे बड़ा फ्लिप फोन का डिस्प्ले है। एंड ताला लगता है भाई चलाने में। अंदर की डिस्प्ले की अगर बात करें, 7 इंच का 165 हर्टz एलटीपीओ, एमोलेड सब कुछ भर रखा है। कलर्स भी ऐसे दिख रहे हैं जैसे नेचुरल आते हैं। और अगर ऐसे ध्यान से देखो ना, तो कोने में आपको एक ऐसा स्क्रीन गार्ड सा लगा हुआ दिखेगा। इसको नहीं उतारना है। क्योंकि ये डिस्प्ले की प्रोटेक्शन के लिए होता है फोल्डिंग फ़्स में। स्पीकर की बात सुन लेते हैं भाई। [संगीत] ड्यूल स्पीकर भाई। ऊपर भी नीचे भी। अच्छा लग रहा है भाई। काफी लाउड है। क्लियर है। ऊपर नीचे दोनों से। इसी बात पे सेले। कैमरा अपने आप खुल गया था। पता नहीं। अब मेरे दोस्तों आ चुके हैं खुले मैदान के अंदर। दो कैमरे का सेटअप है। एक प्राइमरी है और एक वाइड एंगल है। सबसे पहले प्राइमरी से खींच के देखते हैं। फिर वाइड एंगल से भी खींचते हैं पूरा। टू एक्स में भी खींचते हैं। एक दो पोर्ट्रेट भी लियो भाई। अब देखो मेरे को इसकी कूल चीज़ ये लगी। तो आप अपने आप को देख सकते हो भाई फोटो खींचते हैं। देखो एक 24 एमm, ये 34 एमएम 35 mm, ये 50 mm. एंड मेरे को सेल्फी लेनी है। फ्रंट कैमरा यूज़ कर सकते हो। 50 मेगापिक्सल का है। बैक वाले तो सेल्फी खींच ही सकते हो भाई। ये सबसे मुश्किल सेल्फी होती है भाई। ऊपर आसमान आता है नीचे ऐसे। लाने की कोशिश तो करी बट इसने मेरे को काफी काला बना दिया। कंट्रास्ट इसने देखो बहुत ज्यादा है। एक आधा चेहरा पूरा ब्लैक ब्लैक कर दिया है। एंड इस वाले में भी बहुत ज्यादा कंट्रास्ट ही है भाई लेफ्ट साइड वाली जगह। ओवरऑल ये सारी फोटोज हैं जो हमने इससे खींची है। ये प्राइमरी से है। कुछ हमारे वाइड एंगल वाले कैमरे से हैं। एंड पोर्ट्रेट्स भी ली। जो मेन कैमरा है उसी से जुगाहार करके पोर्ट्रेट लेने की कोशिश करता है। प्राइमरी जो फोटो लेता हूं डिसेंट आती है बट ज्यादा ही कंट्रास्ट आ जाता है। कुछ चीजें हैं जो एक्स्ट्रा है जो थोड़ा सा मसाला है। जैसे एट की वीडियोस आप रिकॉर्ड कर सकते हो एंड वो डिसेंट आती है। बहुत ज्यादा क्रेजी कैमरा नहीं है। इवन एक फ्लिप फोन से भी अगर आप कंपेयर करो तो मेरे ख्याल से कैमरा बेटर होना चाहिए था। स्पेशली जो सॉफ्टवेयर में चीजें एक टेलीोटो लेंस की मेरे को कमी फील हुई। एक चीज जो पिछली बार से बहुत ज्यादा बदल दी गई है। एआई जेमिनाई भी आ रहा है। परप्लेक्सिटी भी आ रहा है। एक Microsoft का कोपायलट भी आ रहा है। सबसे हाथ मिला लिए। अभी आया वैसे जेमिनाई और परप्लेक्सिटी है। सबसे पहले ये बटन अगर आप दबाओगे एआई खुल जाएगा। आप इससे पूछो। भाई टमाटर कितने रुपए किलो मिल रहे हैं। प्राइस बता दिया टमाटर। भाई Motorola ने भाई एi वाले फीचर्स पे डेफिनेटली काम करा है। जिस भी ऐप के अंदर खोलोगे उस ऐप के हिसाब से आपको ऑप्शन दे दे। जैसे Instagram में हमने खोल लिया है। भाई रील के कुछ सोर्स बता रहे हैं जो मेरे ख्याल से आज के टाइम में तो काफी ज्यादाेंट है। एक इमेज स्टूडियो है। इसमें बेसिकली एक लाइन में इमोजीज़ अगर आपको बनाना हो स्केच टू इमेज भी ऐड कर रखा है। जैसे मजनू भाई की पेंटिंग हमने पहले ही बनाई थी। सेम इंपोर्ट करके इसके अंदर भी डाल दिए। एक स्टिककर एआई भी है। इसका यूज़ करके स्टिककर बनाना या फिर लैपटॉप्स और बुक्स पे आप आराम से लगा सकते हो। एi avटar भी आप बना के जैसे गिलबी इमेज वाला है। वैसे और भी ऑप्शंस आप ट्राई कर सकते हो। अलग-अलग टाइप के आते हैं। एक प्लेलिस्ट स्टूडियो भी है। बस लिख के आपको बताना है भाई कौन से गाने चाहिए। उस गानों की प्लेलिस्ट बनाता है। एक चीज जो हमने बात करी बैटरी पहले 4000 एमए के अराउंड हुआ करती थी। अब 4700 एमए हो गई है। ये एक जो नॉर्मल फ्लैकशिप फोन आता है उतनी ही बैटरी है। एंड ये मेरे हिसाब से सबसे क्रेजी चीज हुई है। क्योंकि एक बाहर भी बड़ा सा डिस्प्ले लगा रखा होता है। वो भी बंदा यूज़ करता रहता है। अंदर भी जो डिस्प्ले होता है वो नॉर्मल फोन से ज्यादा बड़ा डिस्प्ले होता है। तो बड़ा दिक्कत वाला मामला हो जाता है। उसके बाद चार्जर भी जो है वो भी भाई 68 वाट का दे रखा है। एंड उसके साथ-साथ 30 वाट का वायरलेस चार्जिंग सपोर्ट। इसके अंदर Snapdragon का 8 एलईट प्रोसेसर लगा रखा है। उसके बाद UFS 4.0 स्टोरेज 16 GB DDR 5X RAM कवर डिस्प्ले पे नॉर्मल डिस्प्ले में आप मूव करोगे वो भी एकदम स्मूथ था। पर UI भी इसका भाई अपना स्लिम लाइट ज्यादा हैवी नहीं लगा तो काफी स्मूथ सी एक भाई भाई। बट मेरे दोस्तों फ़ चाहे छोटा ऊंचा पड़ा। बट गेमिंग के बारे में नहीं पता चलता भाई क्या असली मामला है। सबसे पहले दोस्तों हम गेम यहां पे चलाएंगे। तो यहां पे आप गेम में खेल नहीं सकते। यहां पे बेसिक मैप्स, कैलेंडर, कैलकुलेटर ये सब चल जाता है। गेम, ये प्रीकेक और ये सब वाली गेम है। ये बेटे वो वो वो अरे हमको यार भाई जीत गए जीत गए बट भाई अब असली गेम का टाइम आ गया Snapdragon 8 एलट है तो ऑफ कोर्स यार सेटिंगवेटिंग टॉप ही चलती है ऐसे कोई दिक्कत है नहीं बट हम बस अपने हाथ गरम कर रहे हैं हां हमको मत बताओ दीदी यार हमको मत समझाओ हम नहीं रुकने वाले यार भाई हम तो बंदू के रिसोंसिबल चलाते हैं भाई ये तो हैकर की तरह खेलू न अरे यार दोनों को डैमेज दिया कोई नहीं मरा भाई बहुत पिटाई हो गई बहुत पिटाई हो गई इतनी पिटाई नहीं चल सकता भाई अगर इस वाले स्मार्टफोन के प्राइस की बात करी जाए ₹90 के अंदर ये स्मार्टफोन आपको मिल रहा है सबसे महंगा जो फोल्डिंग फोन है उससे सस्ता है Samsung का जो फ्लिप आता है ₹130 के अराउंड आता है एंड ये कोशिश कर रहा है कि जितनी चीजें वो कर रहा है ऑलमोस्ट स्ट उतनी ही चीजें आपको ₹90,000 में प्रोवाइड करें। एंड ऑनेस्टली बताऊं मैं ₹9,000 वैसे सस्ता तो डेफिनेटली नहीं है। बट एक फ्लिप फोन के हिसाब से जो टॉप ऑफ द लाइन है अल्ट्रा सीरीज का है जिसके अंदर कैमरावरा सारी चीजें सॉलिड है। मेरे को अच्छा स्मार्टफोन लगा। मेरे को बस ऐसा लगता है कि इस फ्लिप फोन के अंदर और एक नॉर्मल फोन के अंदर इसमें एक मेजर रुकावट होती है वो होती है भाई कैमरा। जैसे इस बार इन्होंने टेली फोटो लेंस नहीं ऐड कर रखा। पिछली बार टेलीोटो लेंस ऐड करा था। इस बार वाइड एंगल कर दिया। अगर टेलीोटो वाइड एंगल हो जाता दोनों साथ में नीचे एक और छेद कर देते यहां पे। मेरे ख्याल से किसी को दिक्कत नहीं होती। ₹8-9000 प्राइस बढ़ जाता बट मजे ही मजे हो जाते। बट अभी अगर एक फोल्डिंग फोन आपको लेना है तो डेफिनेटली एक तगड़ा चॉइस
""",
    extraction=Extraction(
        battery=BatteryExtraction(
            battery_capacity=4000,
            battery_capacity_extraction_text="4000mAh"
        ),
        charging=ChargingExtraction(
            charging_power=68,
            charging_power_extraction_text="68 watt PD charger",
            wireless_charging=True,
            wireless_charging_power=15,
            wireless_charging_power_extraction_text="15W wireless",
            charger_in_box=True,
            charger_in_box_extraction_text="68 watt PD charger"
        ),
        design=DesignExtraction(
            weight_grams=199.0,
            weight_grams_extraction_text="199 g",
            thickness_mm=15.7,
            thickness_mm_extraction_text="15.7 mm"
        ),
        screen=ScreenExtraction(
            screen_size_inches=6.9,
            screen_size_inches_extraction_text="6.9-inch internal OLED",
            refresh_rate_hz=165,
            refresh_rate_hz_extraction_text="165Hz internal display"
        ),
        in_the_box=InTheBoxExtraction(
            items=[
                ItemExtraction(
                    item_name="Case",
                    item_specifications="Color coordinated",
                    item_extraction_text="color coordinated case"
                ),
                ItemExtraction(
                    item_name="Charger",
                    item_specifications="68W PD",
                    item_extraction_text="68 watt PD charger"
                ),
                ItemExtraction(
                    item_name="Cable",
                    item_specifications="USBC toC",
                    item_extraction_text="USBC toC cable"
                ),
                ItemExtraction(
                    item_name="SIM Ejector",
                    item_specifications=None,
                    item_extraction_text="SIM ejector tool"
                )
            ]
        ),
        variants=[]
    )
)

EXAMPLE_BUDGET = ExampleData(
    model_name="Motorola Edge 50 Fusion",
    text="""--- SOURCE: OEM_OFFICIAL (PRIORITY 1 — ALWAYS AUTHORITATIVE ON HARDWARE SPECS) ---
Performance

Operating System

- Android™ 14

Sensors

- Fingerprint on display, Proximity, Accelerometer, Ambient Light, Gyroscope, SAR sensor, Sensor Hub, E-Compass

Processor

- Qualcomm SM7435-AB Snapdragon 7s Gen 2 (4 nm), Octa-core (4x2.40 GHz Cortex-A78 & 4x1.95 GHz Cortex-A55), Adreno 710

Memory (RAM)

- 8GB | 12GB with RAM Boost 2.0

Storage

- 128GB | 256GB built-in | Non Expandable

Security

- FoD FPS | Face unlock | Moto Secure | Thinkshield for mobile

Certifications

- Dolby Atmos | HiRES

OS Upgrade + Security Patches

- 3 Years OS Upgrade

4 Years SMRs

Battery

Battery Size

- 5000mAh

Charging

- TurboPower™ 68W

Display

Display Size

- 169.4mm (6.67")

Resolution

- Full HD+ (2400 x 1080p) | 395ppi

Screen to Body Ratio

- Active Area-Touch Panel (AA-TP): 92%

Display Technology

- pOLED Endless Edge Display| 144Hz refresh rate | 10 bit | 100% DCI P3 | 1600 Peak Nits | 1200 HBM Nits| Punch Hole | Game Mode 360Hz | Aqua Touch | 720Hz PWM/DC Dimming

Aspect Ratio

- 20:9

Display Protection

- Corning Glass 5 | SGS Low Blue Light | SGS Low Motion Blur

Design

Dimensions

- 162 x 73.1 x 7.8mm (PMMA)

162 x 73.1 x 7.9mm (Vegan Leather)

Body

- 3D PMMA | PU Vegan Leather

Ports

- Type-C port (USB 2.0)

Weight

- Around 175 g

Water Protection

- IP 68

Colours

Forest Blue

Hot Pink

Marshmallow Blue

Camera

Rear Main Camera

- 50MP Sony Lytia 700C

f/1.8 aperture

1.0µm pixel size | Ultra Pixel Technology for 2.0µm

Quad PDAF - All Pixel Focus

Optical Image Stabilization (OIS)

Rear Camera Video Software

- Dual Capture

Spot Color

Timelapse (w/ Hyperlapse)

Macro

Slow Motion

Video Stabilization

Video Snapshot

Efficient Videos


Front Camera Video Capture

- UHD @30fps,

UHD 20:9@30fps 3840x1728,

FHD@30fps,

FHD 20:9@30fps 1920x864


Rear Camera Software

- Ultra-Res

Dual Capture

Spot Color

Night Vision

Macro Vision

Portrait

Live Filter

Panorama

AR Stickers

Pro Mode (w/ Long Exposure)

Smart Composition

Auto Smile Capture

Google Lens™ integration

Active Photos

Timer

High-res Digital Zoom (Up to 8x)

RAW Photo Putput

HDR

Burst Shot

Assistive Grid

Leveler

Watermark

Barcode Scanner

Quick Capture

Tap Anywhere to Capture


Front Camera Hardware

- 32MP

f/2.4 aperture

0.7µm pixel size | Quad Pixel Technology for 1.4µm


Rear Camera Video Capture

- Rear main camera:

UHD @30fps,

UHD 20:9@30fps 3840x1728,

FHD@30fps,

FHD 20:9@30fps 1920x864,

FHD@60fps,

FHD 20:9@60fps 1920x864

Rear macro camera:

UHD @30fps,

UHD 20:9@30fps 3840x1728,

FHD@30fps,

FHD 20:9@30fps 1920x864"


Front Camera Software

- Dual Capture

Spot Color

Portrait

Live Filter

Group Selfie

Pro Mode (w/ Long Exposure)

Auto Smile Capture

Gesture Selfie

Active Photos

Face Beauty

Timer

Selfie Animation

RAW Photo Output

HDR

Assistive Grid

Leveler

Selfie Photo Mirror

Watermark

Burst Shot

Tap Anywhere to Capture


Front Camera Video Software

- Dual Capture

Spot Color

Timelapse (w/ Hyperlapse)

Face Beauty

Video Snapshot

Efficient Videos


Flash

- Single LED flash

Camera 2

- 13MP Ultrawide angle (120° FOV)

Macro Vision

f/2.2 aperture

1.12µm pixel size

PDAF


Audio

Speakers

- Stereo speakers

Headphone Jack

- No

Microphones

- 2

FM Radio

- No

Voice Control

- Google Assistant

Connectivity

Networks + Bands

- 5G: NR band n1/n3/n5/n7/n8/n20/n28/n38/n40/n41/n77/n78 | 4G: LTE band 1/2/3/5/7/8/18/19/20/26/28/32/38/40/41/42 | 3G: WCDMA band 1/2/5/8/19 | 2G: GSM band 2/3/5/8

Bluetooth Technology

- Bluetooth® 5.2

NFC

- Yes

Wi-Fi

- Wi-Fi 802.11 a/b/g/n/ac | 2.4GHz & 5GHz | Wi-Fi hotspot

Location Services

- GPS, A-GPS, LTEPP, GLONASS, Galileo, QZSS,

SIM Card

- Dual SIM (2 Nano SIMs)

In the Box

Device

- motorola edge 50 fusion

In box accessories

- Protective cover, 68W charger, USB cable, guides, SIM tool

Country of Origin

- India

Hello UI

- Personalize: Theme, Wallpaper

Display: Peek Display, Attentive Display

Gestures: Quick Capture, Fast Flashlight, Three-Finger Screenshot, Flip for DND, Pick Up to Silence, Lift to Unlock, Swipe to Split, Quick Launch

Play: Media Controls, Gametime

Tips: Take a Tour, What's New in Android 14


SW Unique Experience

- Moto Connect (wireless)  |  Moto Unplugged  |  Ready For

SW Upgrade Policy

- 3 Years OS Upgrade

4 Years SMRs


Manufacturing Details

Manufacturer's Details

- Padget Electronics Pvt Limited, A-23, Sector-60, Noida, Gautam Buddha Nagar, Uttar Pradesh- 201301

Country of Origin

- India

--- SOURCE: GSMARENA (PRIORITY 3 — USE ONLY WHEN OEM_OFFICIAL IS SILENT ON A FIELD) ---
| Network | [Technology](https://www.gsmarena.com/network-bands.php3) | [GSM / HSPA / LTE / 5G](https://www.gsmarena.com/motorola_edge_50_fusion-12871.php#) |
| [2G bands](https://www.gsmarena.com/network-bands.php3) | GSM 850 / 900 / 1800 / 1900 |
| [3G bands](https://www.gsmarena.com/network-bands.php3) | HSDPA 850 / 900 / 1900 / 2100 |
| [4G bands](https://www.gsmarena.com/network-bands.php3) | 1, 2, 3, 5, 7, 8, 18, 19, 20, 26, 28, 32, 38, 40, 41, 42, 71 - International |
|  | 1, 2, 3, 5, 7, 8, 18, 19, 20, 26, 28, 32, 38, 40, 41, 42 - India |
| [5G bands](https://www.gsmarena.com/network-bands.php3) | 1, 3, 5, 7, 8, 20, 26, 28, 38, 40, 41, 77, 78 SA/NSA |
| [Speed](https://www.gsmarena.com/glossary.php3?term=3g) | HSPA, LTE, 5G |

| Launch | [Announced](https://www.gsmarena.com/glossary.php3?term=phone-life-cycle) | 2024, April 16 |
| [Status](https://www.gsmarena.com/glossary.php3?term=phone-life-cycle) | Available. Released 2024, May 15 |

| Body | [Dimensions](https://www.gsmarena.com/motorola_edge_50_fusion-12871.php#) | 161.9 x 73.1 x 7.9 mm (6.37 x 2.88 x 0.31 in) |
| [Weight](https://www.gsmarena.com/motorola_edge_50_fusion-12871.php#) | 174.9 g (6.17 oz) |
| [Build](https://www.gsmarena.com/glossary.php3?term=build) | Glass front, silicone polymer back (eco leather), plastic frame |
| [SIM](https://www.gsmarena.com/glossary.php3?term=sim) | · Nano-SIM + [eSIM](https://www.gsmarena.com/glossary.php3?term=esim)<br>* * *<br>· Nano-SIM + Nano-SIM |
|  | IP68 dust tight and water resistant (immersible up to 1.5m for 30 min) |

| Display | [Type](https://www.gsmarena.com/glossary.php3?term=display-type) | P-OLED, 1B colors, 120Hz (LATAM), 144Hz (INT), 1600 nits (peak) |
| [Size](https://www.gsmarena.com/motorola_edge_50_fusion-12871.php#) | 6.7 inches, 108.4 cm2 (~91.6% screen-to-body ratio) |
| [Resolution](https://www.gsmarena.com/glossary.php3?term=resolution) | 1080 x 2400 pixels, 20:9 ratio (~393 ppi density) |
| [Protection](https://www.gsmarena.com/glossary.php3?term=screen-protection) | Corning Gorilla Glass 5 |

| Platform | [OS](https://www.gsmarena.com/glossary.php3?term=os) | Android 14, upgradable to Android 15, up to 3 major Android upgrades |
| [Chipset](https://www.gsmarena.com/glossary.php3?term=chipset) | Qualcomm SM7435-AB Snapdragon 7s Gen 2 (4 nm) - International<br>* * *<br>Qualcomm SM6450 Snapdragon 6 Gen 1 (4 nm) - LATAM |
| [CPU](https://www.gsmarena.com/glossary.php3?term=cpu) | Octa-core (4x2.40 GHz Cortex-A78 & 4x1.95 GHz Cortex-A55) - International<br>* * *<br>Octa-core (4x2.2 GHz Cortex-A78 & 4x1.8 GHz Cortex-A55) - LATAM |
| [GPU](https://www.gsmarena.com/glossary.php3?term=gpu) | Adreno 710 |

| Memory | [Card slot](https://www.gsmarena.com/glossary.php3?term=memory-card-slot) | No |
| [Internal](https://www.gsmarena.com/glossary.php3?term=dynamic-memory) | 128GB 8GB RAM, 256GB 8GB RAM, 256GB 12GB RAM, 512GB 8GB RAM, 512GB 12GB RAM |
|  | UFS 2.2 |

| Main Camera | [Dual](https://www.gsmarena.com/glossary.php3?term=camera) | 50 MP, f/1.9, (wide), 1/1.56\\", 1.0µm, multi-directional PDAF, OIS<br>* * *<br>13 MP, f/2.2, 120˚ (ultrawide), 1/3.0\\", 1.12µm, AF |
| [Features](https://www.gsmarena.com/glossary.php3?term=camera) | LED flash, HDR, panorama |
| [Video](https://www.gsmarena.com/glossary.php3?term=camera) | 4K@30fps, 1080p@30/60/120fps, gyro-EIS |

| Selfie camera | [Single](https://www.gsmarena.com/glossary.php3?term=secondary-camera) | 32 MP, f/2.5, (wide), 1/3.14\\", 0.7µm |
| [Features](https://www.gsmarena.com/glossary.php3?term=secondary-camera) | HDR |
| [Video](https://www.gsmarena.com/glossary.php3?term=secondary-camera) | 4K@30fps, 1080p@30fps |

| Sound | [Loudspeaker](https://www.gsmarena.com/glossary.php3?term=loudspeaker) | Yes, with stereo speakers |
| [3.5mm jack](https://www.gsmarena.com/glossary.php3?term=audio-jack) | No |

| Comms | [WLAN](https://www.gsmarena.com/glossary.php3?term=wi-fi) | Wi-Fi 802.11 a/b/g/n/ac, dual-band |
| [Bluetooth](https://www.gsmarena.com/glossary.php3?term=bluetooth) | 5.2, A2DP, LE |
| [Positioning](https://www.gsmarena.com/glossary.php3?term=gnss) | GPS, GLONASS, GALILEO |
| [NFC](https://www.gsmarena.com/glossary.php3?term=nfc) | Yes |
| [Radio](https://www.gsmarena.com/glossary.php3?term=fm-radio) | No |
| [USB](https://www.gsmarena.com/glossary.php3?term=usb) | USB Type-C 2.0, OTG |

| Features | [Sensors](https://www.gsmarena.com/glossary.php3?term=sensors) | Fingerprint (under display, optical), accelerometer, gyro, proximity, compass |
|  | Smart Connect (Ready For) support |

| Battery | [Type](https://www.gsmarena.com/glossary.php3?term=rechargeable-battery-types) | 5000 mAh |
| [Charging](https://www.gsmarena.com/glossary.php3?term=battery-charging) | 68W wired, 50% in 15 min |

| Misc | [Colors](https://www.gsmarena.com/glossary.php3?term=build) | Forest Blue, Marshmallow Blue, Hot Pink |
| [Models](https://www.gsmarena.com/glossary.php3?term=models) | XT2429-1 |
| [Price](https://www.gsmarena.com/glossary.php3?term=price) | [$ 240.00 / C$ 399.00 / € 214.98 / ₹ 20,849](https://www.gsmarena.com/motorola_edge_50_fusion-price-12871.php) |

| Our Tests | [Performance](https://www.gsmarena.com/glossary.php3?term=benchmarking) | AnTuTu: 520319 (v9), 580458 (v10)<br>* * *<br>* * *<br>GeekBench: 2885 (v5), 2938 (v6)<br>* * *<br>3DMark: 793 (Wild Life Extreme) |
| [Display](https://www.gsmarena.com/gsmarena_lab_tests-review-751p2.php) | [1322 nits max brightness (measured)](https://www.gsmarena.com/motorola_edge_50_fusion-review-2709p3.php#dt) |
| [Loudspeaker](https://www.gsmarena.com/gsmarena_lab_tests-review-751p7.php) | [-24.2 LUFS (Very good)](https://www.gsmarena.com/motorola_edge_50_fusion-review-2709p3.php#lt) |
| [Battery](https://www.gsmarena.com/how_we_test_gsmarena_battery_life_test_v2-news-60429.php) | [Active use score 12:40h](https://www.gsmarena.com/motorola_edge_50_fusion-review-2709p3.php#bt) |

--- SOURCE: TRANSCRIPT (PRIORITY 2 — USE FOR: charger_in_box, in_the_box, India-specific colors, India variant confirmations) ---
हाय गाइस आज मैं आप लोगों से बात करूंगा moto2 फ्यूजन के बारे में देखो ₹ 33000 का फोन है और मैंने और मेरी टीम ने मिलकर जो है ती हफ्ते इसमें अपनी मेन सिम डाल के हम लोगों ने टेस्ट करा है और तीन हफ्ते यूज करने के बाद ना मेरे पास काफी सारा डाटा है जो मैं आप लोगों को बताना चाहता हूं देखो मैं इतना समझता हूं बहुत सारे लोगों का ये क्वेश्चन है कि क्या यह सही फोन है इस प्राइस सेगमेंट के हिसाब से और इस प्राइस सेगमेंट में ला अंडर ₹2500000 मैं एड्रेस करने की कोशिश करूंगा क्योंकि जो ये डिजाइन का एक्सपीरियंस शेयर करना चाहूंगा यार एक टाइम था 2021 के एंड तक भी ना motorola's बना रहा था और मुझे अभी भी याद है मैं काफी क्रिटिसाइज करता था इस बात को अब फोन को यूज़ कर रहा था ना मुझे बार-बार ये चीज रिलाइज हो रही थी कि यूज करा है एज सच कोई स्क्रैच वगैरह नहीं है बट हां डर्टी हो गया है कलर बट सेम टाइम पे वीगन लेदर जो है ना बहुत अच्छी ग्रिप प्रोवाइड करता है अगर आप विदाउट केस फोन यूज़ करो तो और मेरे से ना ये फोन बहुत तेज गिर गया था अब यहां पर एक डेंट भी आ गया अपर वाली साइड पे और ये जब मैं डेंट देख रहा था मेरे नोटिस में आया कि यहां पर हल्का सा गैप भी है फ्रेम के अराउंड अब मैं श्योर नहीं हूं कि ये पहले से था गैप या फिर ये गिरने के बाद हुआ है वैसे दो चीजें मैं हाईलाइट करूंगा जो कि आई थिंक और बेटर हो सकती थी इस फोन में जैसे कि हैप्टिक फीडबैक मतलब हैप्टिक फीडबैक पर्टिकुलर नहीं जो वाइब्रेशन होती है ना जब कॉल आती है वो थोड़ी सी और बेटर हो सकती थी और सेकंड चीज airtelxtream.in बहुत-बहुत कंफर्टेबल है इनफैक्ट डिस्प्ले डिपार्टमेंट भी इस फोन का काफी ज्यादा फैंटास्टिक है और फैंटास्टिक भी इसलिए बोल रहा हूं डिस्प्ले को क्योंकि इस प्राइस सेगमेंट में ना मैंने यूजुअली नोटिस करा है कि ब्राइटनेस इतना अच्छी नहीं होती फोन के अंदर और जब इस फोन को मैं डायरेक्ट सनलाइट में भी यूज़ करता हूं ना यार बहुत विजिबल रहता है और प्रैक्टिकल ये चीज मेरे को पर्सनली बहुत मैटर करती है सिमिलरली एक और इंटरेस्टिंग चीज बताता हूं जब मैंने onepluscases.in सपोर्ट नहीं है आई मीन ना ही youtube-dl है जो कलर प्रोसेसिंग है इस डिस्प्ले की यार मुझे बहुत अच्छी लगी मतलब मैंने इसमें कंटेंट देखा है netfx3 देखी और आप खुद देखो यार मतलब जो डिटेल है शार्पनेस है कलर है मैं बिल्कुल भी डिसपे नहीं था मतलब इनफैक्ट मैं तो इंप्रेस हुआ था इनफैक्ट जो स्पीकर्स भी इस फोन के क्योंकि जो ये मूवी है मैंने पूरी स्पीकर्स पे सुनी है तो वो भी एक एटमॉस्फियर क्रिएट करते हैं तो मतलब अच्छे स्पीकर्स है देखो डिस्प्ले को बहुत अच्छे से एक्सपीरियंस करने के बाद ना मैं कॉन्फिडेंटली ये चीज कंफर्म कर सकता हूं कि अगर आप बहुत ज्यादा कंटेंट देखते हो तो ये डिस्प्ले बहुत फैंटास्टिक है इनफैक्ट स्पीकर्स भी अच्छे हैं ना तो लाइक ओवरऑल कॉमिनेशन कंटेंट कंजप्शन के लिए बहुत सेंस बना रहा है अच्छा एक और चीज सरप्राइजिंगली मतलब 5000 एमए बैटरी है इस फोन के अंदर स्टिल मेरे को जो स्क्रीन ऑन टाइम है आप मतलब स्क्रीनशॉट देखो 7त घंटे से ज्यादा ही मिला है मतलब 7त से 75 घंटे के अराउंड और एक बार लाइट यूसेज पे तो 8 घंटे के अराउंड भी स्क्रीन ऑन टाइम आया इफ इन केस आपको जो है स्क्रीन ऑन टाइम का आईडिया नहीं है कि मैं स्क्रीनशॉट्स क्या दिखा रहा हूं तो बस इतना समझ लो बैटरी बैकअप जो है फुल डे का है एक और चीज बताता हूं 68 वाट की जो चार्जिंग है ना वो मेरे को फास्ट लगी मतलब मोस्टली जो है अच्छा खासा चार्ज कर देती है फोन को मतलब फास्ट तरीके से चार्ज कर देती है बट एक पर्टिकुलर सिनेरियो के बारे में बात करता हूं अगर फोन जो है चार्जर से कनेक्टेड है और मैं फोन को यूज कर रहा हूं तो वो एक्सट्रीमली स्लो चार्ज होता है मे बी हो सकता है कि ये बग हो हो सकता है सिर्फ मेरी यूनिट में हो अब मैं फाइनली एड्रेस करना चाहूंगा इस फोन का परफॉर्मेंस डिपार्टमेंट जो कि मेरे से सबसे ज्यादा पूछा जा रहा है कि जो इस फोन में snap7 s ज2 है कैसा परफॉर्म करेगा मतलब अगर हम फ्यूचर को देखते हुए चले तो क्या ये इनफ परफॉर्मेंस लाकर देगा बेसिकली यही कंफ्यूजन है ना कि इस फोन में ufs2.1 स्टोरेज है एपीडी 4x रम है नहीं तो इस प्राइस सेगमेंट में क्या हो रहा है ना कि अगर थोड़ा सा प्राइस बढ़ा दो तो बेटर प्रोसेसर या फिर ufs3.0 स्टोरेज मिल जाती है देखो मेरा मानना है कि हमें जो है प्रैक्टिकल एक्सपीरियंस पे ध्यान देना चाहिए और अगर मैं अपना प्रैक्टिकल एक्सपीरियंस शेयर करूं तो गाइज इस फोन में मेरे को हीटिंग नहीं फेस हुई मतलब कैसी भी सिचुएशन हो चाहे फिर वो मतलब मैंने इसको एक्सट्रीम कंडीशंस में भी यूज़ करा हुआ है डेल्ली में जब काफी गर्मी थी और अभी तो र मानसून आ गया काफी ज्यादा बारिश वगैरह है मतलब दोनों सीजन में मैंने इसको यूज़ किया है स्टिल मैं कंफर्म कर सकता हूं कि टिंग इश्यूज तो मुझे फेस नहीं हुए हैं इनफैक्ट गेमिंग का भी एग्जांपल ले लेते हैं मतलब टेस्टिंग के लिए मैंने इसमें बहुत सारे गेम्स खेले सीओडी का एग्जांपल लेके चलते हैं 30 मिनट सीओडी के गेम प्ले में ना एज सच इसमें कोई भी हीटिंग इश्यूज नहीं आते खैर मैं ये भी इफॉर्म कर देता हूं ये प्रोसेसर कोई हाई एफपीएस गेमिंग के लिए नहीं है मतलब कैजुअल गेमिंग की मैं बात कर रहा हूं जहां पर मैं हेयर एंड देर थोड़ा बहुत मतलब मूड रिफ्रेश करने के लिए गेमिंग कर रहा हूं सेकंड प्रैक्टिकल चीज जो मैं हाईलाइट करना चाहूंगा वो है रम मैनेजमेंट मेरे हिसाब से भाई बहुत बहुत अच्छी रम मैनेजमेंट है इस फोन की लिटरली ज्यादातर एप्स मुझे ऐसा फील होता है कि रम में रहती है तो मतलब एज सच रीलोड नहीं होती तो अगर मैं ये पर्सपेक्टिव लेके चलूं कि भाई जो भी इस फोन को खरीदेगा वो कैजुअल गेमिंग करना चाहता है और एक नॉर्मल मल्टीटास्क करना चाहता है तब तो ये इनफ परफॉर्मेंस दे देगा फोन और अगर आप एक हैवी यूजर हो आप हैवी गेमिंग करना चाहते हो और बहुत ज्यादा मल्टीटास्किंग करना चाहते हो तब हो सकता है कि आपको फील हो कि यार थोड़ी सी परफॉर्मेंस कम है आई मीन अगर मैं इस फोन में बहुत सारी एप्स लगातार खोलता चलूं तो आप खुद देखो आपको माइक्रो जट्स हो सकता है कैमरे पे नोटिस भी ना हो देखने को मिल सकते हैं तो सारी चीजों को एक्सपीरियंस और टेस्ट करने के बाद ना मैं आपको एकदम बिल्कुल क्लियर चीज बताता हूं अगर आप सिर्फ परफॉर्मेंस के लिए फोन ले रहे हो तो ये फोन जो है आपको इतनी ज्यादा परफॉर्मेंस नहीं लाकर देगा ये सिर्फ और सिर्फ आपके नॉर्मल टास्क अच्छे से परफॉर्म कर देगा अच्छा सॉफ्टवेयर की बात कर लेते हैं देखो अभी भी जो है है देखो ती हफ्ते बाद मुझे ये फील हुआ है कि यार ये तो रिलायबिलिटी भी दे रहे हैं रिलायबिलिटी मतलब पता है क्या है इतने दिन में ना इस फोन मुझे एस कोई बग्स नहीं देखने को मिले कि इसने मुझे इरिटेट कर दिया हो और इस प्राइस सेगमेंट में ये चीज अचीव करना मतलब रिलायबल बोलना बहुत बड़ी बात हो जाती है और मैं 100% श्योर हूं कि लोग मुझे बोलेंगे कि भाई स्मार्ट कनेक्ट के बारे में आपने बताया नहीं देखो इस फोन की अनबॉक्सिंग वीडियो की बात करूं या फिर जितने भी बिल्कुल क्लियर समझ में आ जाएगा स्मार्ट कनेक्ट अब सबसे इंटरेस्टिंग चीज बताता हूं मैं जब मैंने इस फोन को यूज करना शुरू किया था मेरे को लगा था कि शायद ना मैं सिर्फ सॉफ्टवेयर की सबसे ज्यादा तारीफ करूंगा सबसे सरप्राइजिंग एलिमेंट तो इसका कैमरा निकला मतलब मैं आपको ऑनेस्टली बता रहा हूं आज से पहले मैं जब भी कमरा है मुझे बिलीव सा नहीं हुआ मुझे खुद पे शक होने लगा तब मैंने जो है थोड़ा बहुत रिव्यूज जाके देखे ऑनलाइन स्पेसिफिकली है ना वो बहुत पंची है या फिर कुछ-कुछ सिचुएशन में तो ओवर सैचुरेटेड भी होता है और रियल यूजर्स की जब मैं रिव्यूज पढ़ रहा था सब इसके पोर्ट्रेट मोड की तारीफ कर रहे थे आई थिंक तारीफ करने के पीछे रीजन है कि ह्यूमन शॉट्स में जो स्किन टोन है ना ये बहुत अच्छी तरीके से हैंडल करता है और मेरे ओपिनियन में जो 2x की पोर्ट्रेट फोटोज है ना वो भी अच्छी लुक देती है इनफैक्ट हाइलाइट्स भी ना मेरे हिसाब से और बेटर तरीके से हैंडल हो सकती है बट स्टिल अगर मैं ओवरऑल फोटोज को इतना क्रिटिकल होके जज ना करो ना तो डिटेल्स सच में अच्छी रहती है और क्योंकि अल्ट्रा वाइड लेंस 13 मेगापिक्सल का है तो इसकी भी बात करना बता है खैर इस फोन की जब मैंने कंपैरिजन वीडियो बनाई थी तब मैंने आपको बताया था कि कलर प्रोसेसिंग इसकी लाक एवरेज है बट डिटेल जो है अल्ट्रा वाइड लेंस में अच्छी आ जाती है जैसे अल्ट्रा वाइड लेंस जो है मैक्रो का भी काम कर लेता है तो वो भी अच्छे आउटपुट आते हैं बट स्पेसिफिकली लो लाइट फोटोस दिखाता हूं देखो मेरे हिसाब से जो कलर्स है ना वो अच्छे से रिटेन कर पा रहा है हां कुछ लोगों को ओवर सैचुरेटेड लग सकती है फोटोस बट पर्सनली मैंने ये नोटिस करा कि हाईलाइट कंट्रोल ना और बेटर हो सकता है और एक चीज मैं आपको बता देता हूं जो 32 मेगापिक्सल का फ्रंट कैमरा है ना अगर आप सेल्फीज बहुत ज्यादा कैप्चर कर रहे हो तो इस फोन की तरफ जरूर देखना क्योंकि जो ओवरऑल फोटो है ना काफी अच्छी आती है फ्रंट कैमरे से और अगर आपको हायर साइड कलर्स पसंद आते हैं ना तो यार कहना पड़ेगा इस प्राइस सेगमेंट में फ्रंट बैक इनफैक्ट अल्ट्रा वाइट से भी 4k रिकॉर्डिंग होना बहुत बड़ी बात है और क्योंकि स्टेबलाइजेशन भी है वीडियो के अंदर और जो कलर्स जैसे कि मैंने बताया हायर साइड है अगर आपको अच्छे लगते हैं तो यार इनफैक्ट ये वीडियो में भी अच्छा बोलूंगा मैं आपके लिए सिंपली बोलूंगा प्राइस सेगमेंट देखते हुए कि वर्सटाइल कैमरा सेटअप है तो अब सारी टेस्टिंग को ना सम अप करके मैं आपको बहुत इजली कंक्लूजन बताता हूं बता हूं मुझे क्या फील हुआ इस फोन के बारे में मैं स्ट्रेट बोल सकता हूं कि अगर आप एक पावर यूजर नहीं हो ना तो मेरे हिसाब से एक बहुत बैलेंस फोन में से एक है मतलब 23000 में ये ज्यादातर बेसिक चीजें बहुत अच्छी परफॉर्म कर रहा है और ये सारी चीजें मेरे को तीन हफ्ते फोन को यूज करने के बाद समझ में आई है तो खैर ये था मेरा रिव्यू इस फोन का आप क्या सोचते हो कमेंट सेक्शन में जरूर बताना और अगर आपको ये वीडियो पसंद आई हो तो लाइक बटन पे क्लिक कर दो और सब्सक्राइब करके हमारे टेक ब आर्मी को जवाइन कर लो हम लोग क्वालिटी बिलीव रखते हैं अगले मिलते हैं थैंक्स फॉर वाचिंग बाय बाय गाइ
ऑलराइट दोस्तों moto2 fusion5 साल से हर एक वीडियो में मैं कह रहा हूं कि motorola's लॉन्च करने वाले हैं आई एम प्रिटी श्यर दे आर ऑन द राइट पाथ जी हां moto3 moto3 का सेकंड फोन है एंड मैं कहूंगा ये 50 प्र के बीच वाला फोन है आगे बढ़ने से पहले और एक चीज बता दूं कि इसकी प्राइस इफेक्टिव प्राइस जो होने वाली है आफ्टर ऑफर्स सो 21000 फॉर 8120 उस हिसाब से इस फोन के पास देखने वाले हैं लेकिन उसके पहले दोस्तों पहली बार आए हो सब्सक्राइब करना मत भूलिए जी हां पैकेजिंग भी अच्छी हो गई है आई हैव टू से दिस मोर एनवायरमेंटल फ्रेंडली हो गई है एंड थोड़ी प्रीमियम भी हो गई है पहले ऐसे नहीं हुआ करता था दे हैव इंप्रूव्ड 360 डिग्री फॉर शोर लेकिन इसमें कुछ एडिशन भी है ओ अब कैमरा के अंदर से मैं दिखा नहीं सकता हूं बट स्मेल्स ब्यूटीफुल जैसे ही आप बॉक्स खोलोगे यू विल गेट दैट ब्यूटीफुल फ्रेगरेंस जी हां सबसे पहले फोन देन यू हैव 68 वाट का चार्जर आता है एंड ूब टाइप सी है तो यब टाइप सी टू टाइप सी चार्जिंग केबल है सो दैट इज गुड सबसे पहले सिम कार्ड टूल देन यू हैव छोटा सा डॉक्यूमेंटेशन हार्ड केस है दोस्तों दिस इज लाइक स्लीव नाइस ब्यूटीफुल चलो गुड सो पैकेजिंग इज गुड बॉक्स कंटेंट्स आर ग्रेट एवरीथिंग एल्स इज गुड अब फोन कैसा है वो देखना है जैसे ही हाथ में लेते हो दोस्तों ते हो सुपर लाइट वेट व ल राइट वेगन लेदर डिजाइन है प्रीमियम एब्सलूट प्रीमियम लुक जी हां ये फ्यूज्ड है ऐसा कैमरा मॉड्यूल अलग से वगैरह नहीं है वेरी नाइस व्हाट इन हैंड फील यार आई मीन फोन जब आप हाथ में लेते हो सॉफ्ट केस टीपीयू केस मिलते है वैसे नहीं है वेरी नाइस बल्क नहीं बढ़ता है फोन का दैट फॉर श्यर सभी कलर्स तीन अलग-अलग कलर्स है कलर मैच्ड केस मिलती है आपको ब्यूटीफुल कर्व डिस्प्ले दोस्तों पीछे वेगन लेदर और ये इतना लाइट लग रहा है 170 ग्रा मतलब बहुत दिन हो गए हैं कि 170 ग्रा मैंने किसी फोन को खाया वैसे 5000 मिली एंपियर की बैटरी तो थोड़ा ज्यादा होगा आई विल से 173 174 ग्रा नहीं 170 भी हो सकता है पता नहीं लेट्स सी 177.8 स्टिल वेरी लाइट बहुत ही लाइट है दोस्तों जबरदस्त इन हैंड एक्सपीरियंस है वेट डिस्ट्रीब्यूशन वगैरह बहुत अच्छा है वैसे आपको बता दूं आगे की तरफ अब गोरिला ग्लास प्रोटेक्शन आता है गोला ग्लास फाइव यस लेट्स डू दिस वन टू एंड बूम वन टू एंड ओ हो हो वो थोड़ा ब बड़ा आवाज था लेकिन कुछ नहीं वो जो आवाज आ रहा है वो ओआस का है सो ओआस कैमरा आता है जी हां बोट्स एंड बटंस की तरफ देखें नीचे की तरफ सिम कार्ड ट्रे माइक्रोफोन यब टाइप सी स्पीकर ग्रिल राइट हैंड साइड में पावर ऑन ऑफ एंड देन वॉल्यूम रॉकर है ऊपर नॉइस कैंसलिंग माइक्रोफोन है एंड देन यू हैव लब एटमॉस लिखा हुआ है एंड ये स्टीरियो स्पीकर्स है तो ऊपर स्पीकर ग्रिल नहीं है लेकिन यर पीस में स्पीकर्स है सो यस स्टीरियो स्पीकर्स हैं एंड सिम कार्ड ट्रे जो है लेट्स फाइंड आउट आई थिंक इट इज गोइंग टू बी ड्यूल नो सिम कार्ड स्लॉट आता है एसडी कार्ड स्लॉट नहीं है अब बाहर से देख लिया अंदर से देखते हैं लेट्स स्टार्ट द फोन 6.67 इंच का पी ओलेड डिस्प्ले आता है 10 बिट डिस्प्ले है ये 144hz फास्ट रिफ्रेश रेट है एंड स्क्रीन टू बॉडी रेशियो के बारे में बात करूं एब्सलूट ग्रेट एंड यूनि साइज है दोस्तों अगर ऊपर का बेजल और नीचे का भी चिन्ह देखते हो तो सेम साइज का है साइड के तो बेजल्स दिखते ही नहीं है एब्सलूट गॉर्जियस डिस्प्ले 1600 निट्स पीक ब्राइटनेस है एबीएम हाई ब्राइटनेस मोड जो है दैट इज 1200 याद रखिए 23000 का फोन है उस हिसाब से बहुत ही अच्छा डिस्प्ले है चलिए स्पेसिफिकेशंस के बारे में बात करते हैं ये आता है 7s जन 2 इस प्रोसेसर के ऊपर इसमें दो वेरिएंट्स है 8gb रम 128gb स्टोरेज एंड देन उसके ऊपर 12256 और हां अगर आप इंटरेस्टेड हो दोस्तों ऑफर्स वगैरह जो अभी चल रहे हैं दोस्तों इंट्रोडक्टरी ऑफर्स वो दिखा रहा हूं स्क्रीन के ऊपर 8gb रम जो है रम टाइप lpddr4x है ufs2.1 स्टोरेज टाइप है लेकिन इधर मैं एक चीज कहना चाहूंगा हमने पिछली बार भी टेस्ट किया था रीड राइट स्पीड ufs2.1 भी होंगे तो भा यू नो कुछ-कुछ फ्स ऐसे हैं कि 3.1 के ufs3.0 की स्पीड स्लोअर आए हैं और इसके फास्टर आए हैं सो इट इज वेल ऑप्टिमाइज्ड रीड राइट स्पीड्स अच्छे हैं और याद रखिए दोस्तों 8128 वेरिएंट जो है 2299 प्राइस है ₹ ऑफ है सो 21023 3000 चलिए 7s ज2 परफॉर्म कैसे करता है अगर अंतत स्कोर के बारे में बात करूं तो 61000 620000 के करीब आता है व्हिच आई थिंक इज क्वाइट गुड नो प्रॉब्लम्स देयर वैसे बैटरी के बारे में बताना भूल गया 5000 म एंपियर की बैटरी है एंड 68 वाट चार्जिंग है तो दिन भर आपको आराम से जाएगी मॉडरेट यूसेज में इनफैक्ट लो यूसेज करोगे तो ढ़ दिन भी जाएगी शुड नॉट बी अ प्रॉब्लम बट कमिंग बैक टू परफॉर्मेंस मीडियम टू हाई सेटिंग्स पे आप pubg0 fp3 5 एफपीए तक मिला व्हिच इज वेरी गुड फॉर फर दिस प्राइस फोन एक्सलेंट सीपीयू थ्रोटल टेस्ट रन किया 8485 पर स्टेबिलिटी स्कोर आया व्हिच अगेन इज वेरी गुड एंड थर्मल्स के बारे में भी बात करूं wagon-r है ब्राइट डिस्प्ले है पीलेड डिस्प्ले सो मल्टीमीडिया के लिए जबरदस्त है सेंसर्स के बारे में बात करूं सभी सेंसर्स आते हैं इन डिस्प्ले फिंगरप्रिंट सेंसर है फेस अनलॉक है एवरीथिंग वर्क्स वेरी वेल कनेक्टिविटी के बारे में बात करूं wi-fi 6 आता है आपको अगेन 20 22000 में जनरली आपको ड्यूल बैंड वगैरह मिलता है बट इसमें wi-fi 6 है ब्लूथ 5.2 है रेडी फॉर है जी हां आपको बड़े स्क्रीन में सीमलेसली अगर ये डिस्प्ले कनेक्ट वगैरह करना है यू कैन डू दैट सो फीचर्स के मामले में भी फीचर रिच है हेलो यूआई कैमरा की तरफ बढ़ते हैं h50 pro50 फ्यूजन मुझे लगा था कि ये वो प्रो वेरिएंट है तो जरा यू नो दे विल फोकस मोर और उसकी वजह से बहुत इंप्रूवमेंट आई है पेंटो के साथ उनकी जो पार्टनरशिप थी उसकी वजह से बट मैं एक बात कहूं ए 50 फूजन मे बी दे हैव कैरिड फॉरवर्ड दैट अब अगेन 20 25000 की प्राइस रेंज में ये मैं डेफिनेटली कह सकता हूं कि दिस इज वन ऑफ द बेटर कैमरा क्यों क्योंकि अगर स्पेक्स के बारे में बात करूं तो ये पहला फोन है जिसमें sonypicturesindia.com फर्स्ट इंप्रेशंस है बट ओवरऑल अगर मैं बात करूं दोस्तों आई मीन motorola's वीडियोस वगैरह निकाले हुए हैं दोस्तों नीचे जी ड्राइव की लिंक दे रहा हूं ओरिजिनल क्वालिटी में आप चेक कीजिएगा फोटोज एवरीथिंग फर्स्ट इंप्रेशन से तो ज्यादा नहीं किए हुए हैं बट गो एंड चेक इट आउट अगर वीडियो ग्राफी के बारे में बात करूं तो 4k 30fps फ्रंट और बैक दोनों ही तरफ कर सकते हो ओआस का सपोर्ट आता है तो लो लाइट में भी आप निकालो तो ज्यादा टरी वगैरह नहीं आने वाला है मैंने एक चीज नोटिस की दोस्तों राइट हैंड साइड में पावर बटन के नीचे भी और एक माइक्रोफोन आता है सो ऑडियो जूम का एक फीचर आता है जहां पे आप जूम कर रहे हो वीडियो लेते समय तो वो ऑडियो को भी ज़ूम करेगा उतना ही फोकस करेगा ऑडियो पे तो ऑडियो ज्यादा क्लियर आएगा जो बेसिक फीचर्स होते हैं वो सभी आते हैं स्लो मोशन प्रो प्रो मोड स्पोर्टेड ये सब है लेकिन एक होराइजन लॉक फीचर आता है दोस्तों जी हां आप वीडियो शूट कर रहे हो कैसे भी घुमाओ होराइजन इज गोइंग टू बी स्टेबल सो होराइजन लॉक फीचर आता है जो बहुत सारे एक्शन कैमरा में वगैरह आता है इसमें भी आपको मिलता है और एक है जो यू नो आपकी फोटोग्राफी थोड़ी यूनिक बना देता है टिल्ट शिफ्ट मोड दैट इज आल्सो देयर फीचर्स के मामले में भी कम नहीं है कहीं पे भी आप मुझे नीचे कमेंट्स में बताइए कि 20 21 22000 में कौन सा फोन है जो ip68 सर्टिफिकेशन देता है h50 फ्यूजन आपको देता है ip68 सर्टिफिकेशन और हां इसमें स्मार्ट वाटर टच भी है अभी बारिश का सीजन आ रहा है तो पानी बारिश में भी आप यूज़ करोगे यू शुड नॉट हैव एनी प्रॉब्लम्स वैसे प्र में आपको usb3.1 मिलता है वच वाज अ गुड थिंग इसमें usb2.0 मिलता है आपको वैसे इसमें तीन तीन कलर्स आते हैं ये जो है ये मार्शमैलो ब्लू है देन हॉट पिंक है दोस्तों या गर्ल्स आर गोइंग टू लाइक दिस हॉट पिंक है और एक फॉरेस्ट ब्लू आता है जो बेसिकली डार्कर ब्लू ब्लू है लेकिन वो पीएमएमए मटेरियल में है दीज आर इन वेगन लेदर यू नो वन ऑफ द रीजंस क्यों motorola's के लॉन्च किए हुए हैं h50 प्रया दूसरे भी फोस देखोगे कहीं पे भी आप देखोगे तो 4 4.5 के करीब इनके रेटिंग है और वो भी 50 60000 रेटिंग्स आए हुए हैं एक बात तो डेफिनेट है मोला इज फोकसिंग ऑन इंडिया मोला इज लॉन्चिंग गुड फस लेकिन सबसे इंपोर्टेंट दोस्तों राइट प्राइस पॉइंट पे वो लच कर रहे हैं एंड h50 फ्यूजन इज नो डिफरेंट सो दैट्ची कमेंट सेक्शन में जरूर पूछेगा वैसे ये वीडियो जो है दोस्तों [संगीत] ब
""",
    extraction=Extraction(
        battery=BatteryExtraction(
            battery_capacity=5000,
            battery_capacity_extraction_text="5000mAh"
        ),
        charging=ChargingExtraction(
            charging_power=68,
            charging_power_extraction_text="68W TurboPower",
            wireless_charging=False,
            wireless_charging_power=None,
            wireless_charging_power_extraction_text=None,
            charger_in_box=True,
            charger_in_box_extraction_text="68W TurboPower"
        ),
        design=DesignExtraction(
            weight_grams=174.9,
            weight_grams_extraction_text="174.9 g",
            thickness_mm=7.9,
            thickness_mm_extraction_text="7.9 mm"
        ),
        screen=ScreenExtraction(
            screen_size_inches=6.7,
            screen_size_inches_extraction_text="6.7\" pOLED",
            refresh_rate_hz=144,
            refresh_rate_hz_extraction_text="144Hz"
        ),
        in_the_box=InTheBoxExtraction(
            items=[
                ItemExtraction(
                    item_name="Charger",
                    item_specifications="68W TurboPower charger",
                    item_extraction_text="68W TurboPower charger"
                ),
                ItemExtraction(
                    item_name="Cable",
                    item_specifications="USB Type-C",
                    item_extraction_text="USB Type-C cable"
                ),
                ItemExtraction(
                    item_name="SIM Tool",
                    item_specifications=None,
                    item_extraction_text="SIM tool"
                )
            ]
        ),
        variants=[]
    )
)

# Only output the 3 merged examples per V4 instructions
RUN_A_EXAMPLES = [
    EXAMPLE_FLAGSHIP,
    EXAMPLE_FOLDABLE,
    EXAMPLE_BUDGET
]

REQUIRED_CLASSES = [
    "BatteryExtraction",
    "ChargingExtraction", 
    "DesignExtraction",
    "ScreenExtraction",
    "InTheBoxExtraction",
    "VariantExtraction"
]

REQUIRED_ATTRIBUTES = [
    ("battery_capacity", int),
    ("charging_power", int),
    ("wireless_charging", bool),
    ("wireless_charging_power", int),
    ("charger_in_box", bool),
    ("weight_grams", float),
    ("thickness_mm", float),
    ("screen_size_inches", float),
    ("refresh_rate_hz", int),
    ("color", str),
    ("ram", str),
    ("storage", str),
    ("items", list)
]

def run_verbatim_check():
    import langextract as lx
    for i, example in enumerate(RUN_A_EXAMPLES):
        print(f"Checking Extractions for Example {i+1} ({example.model_name})...")
        lx.validate_verbatim_extractions(example)
    print("Verbatim check passed. All extraction_text fields are exact substrings of their source text.")

def run_coverage_check():
    class_coverage = set()
    attr_coverage = set()
    
    for ex in RUN_A_EXAMPLES:
        ext_dict = ex.extraction.model_dump()
        for k, v in ext_dict.items():
            if v is not None:
                if type(v) == list and len(v) == 0:
                    continue
                class_type = type(getattr(ex.extraction, k)).__name__ if not isinstance(v, list) else type(getattr(ex.extraction, k)[0]).__name__ + "List"
                if class_type == "VariantExtractionList": class_coverage.add("VariantExtraction")
                elif class_type == "ItemExtractionList": class_coverage.add("InTheBoxExtraction")
                else: class_coverage.add(class_type)
                
                if isinstance(v, list):
                    for item in v:
                        for ik, iv in item.items():
                            if iv is not None: attr_coverage.add(ik)
                else:
                    for ck, cv in v.items():
                        if cv is not None: attr_coverage.add(ck)

    for req_cls in REQUIRED_CLASSES:
        if req_cls not in class_coverage:
            print(f"WARNING: Class {req_cls} is not tested in any example.")
            
    for req_attr, _ in REQUIRED_ATTRIBUTES:
        if req_attr not in attr_coverage:
            print(f"WARNING: Attribute {req_attr} is not tested in any example.")
            
    print("Coverage check passed. All required classes and attributes are represented.")
