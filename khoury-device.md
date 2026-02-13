Linux explorer-02 5.14.0-362.13.1.el9_3.x86_64 #1 SMP PREEMPT_DYNAMIC Wed Dec 13 14:07:45 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         46 bits physical, 57 bits virtual
  Byte Order:            Little Endian
CPU(s):                  96
  On-line CPU(s) list:   0-95
Vendor ID:               GenuineIntel
  Model name:            Intel(R) Xeon(R) Gold 5318Y CPU @ 2.10GHz
    CPU family:          6
    Model:               106
    Thread(s) per core:  2
    Core(s) per socket:  24
    Socket(s):           2
    Stepping:            6
    CPU max MHz:         3400.0000
    CPU min MHz:         800.0000
    BogoMIPS:            4200.00
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mc
                         a cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss 
                         ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art
                          arch_perfmon pebs bts rep_good nopl xtopology nonstop_
                         tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cp
                         l vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dc
                         a sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer 
                         aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpu
                         id_fault epb cat_l3 invpcid_single intel_ppin ssbd mba 
                         ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority e
                         pt vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 
                         erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap
                          avx512ifma clflushopt clwb intel_pt avx512cd sha_ni av
                         x512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc 
                         cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_de
                         tect wbnoinvd dtherm ida arat pln pts vnmi avx512vbmi u
                         mip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_
                         vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid fsrm
                          md_clear pconfig flush_l1d arch_capabilities
Virtualization features: 
  Virtualization:        VT-x
Caches (sum of all):     
  L1d:                   2.3 MiB (48 instances)
  L1i:                   1.5 MiB (48 instances)
  L2:                    60 MiB (48 instances)
  L3:                    72 MiB (2 instances)
NUMA:                    
  NUMA node(s):          2
  NUMA node0 CPU(s):     0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,
                         40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,7
                         6,78,80,82,84,86,88,90,92,94
  NUMA node1 CPU(s):     1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,
                         41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,7
                         7,79,81,83,85,87,89,91,93,95
Vulnerabilities:         
  Gather data sampling:  Mitigation; Microcode
  Itlb multihit:         Not affected
  L1tf:                  Not affected
  Mds:                   Not affected
  Meltdown:              Not affected
  Mmio stale data:       Mitigation; Clear CPU buffers; SMT vulnerable
  Retbleed:              Not affected
  Spec rstack overflow:  Not affected
  Spec store bypass:     Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:            Mitigation; usercopy/swapgs barriers and __user pointer
                          sanitization
  Spectre v2:            Mitigation; Enhanced / Automatic IBRS, IBPB conditional
                         , RSB filling, PBRSB-eIBRS SW sequence
  Srbds:                 Not affected
  Tsx async abort:       Not affected
               total        used        free      shared  buff/cache   available
Mem:           250Gi        30Gi       8.8Gi       4.0Gi       217Gi       220Gi
Swap:           19Gi        12Gi       6.9Gi
Filesystem                                   Size  Used Avail Use% Mounted on
devtmpfs                                     4.0M     0  4.0M   0% /dev
tmpfs                                        126G  7.7G  118G   7% /dev/shm
tmpfs                                         51G  4.0G   47G   8% /run
/dev/sda4                                     20G  4.0G   16G  21% /
/dev/sda2                                    2.0G  313M  1.7G  16% /boot
/dev/sda3                                     30G   14G   16G  48% /var
/dev/sda11                                   337G   25G  313G   8% /tmp
/dev/sda6                                    9.8G  102M  9.7G   2% /srv
/dev/sda1                                   1022M  7.0M 1015M   1% /boot/efi
/dev/sda9                                    9.8G  6.2G  3.6G  64% /var/tmp
/dev/sda7                                    9.8G  705M  9.1G   8% /var/log
/dev/sda8                                    9.8G  139M  9.6G   2% /var/log/audit
tmpfs                                         26G  4.0K   26G   1% /run/user/0
vast1-mghpcc-eth.neu.edu:/vast_shared         30T   22T  8.2T  73% /shared
vast1-mghpcc-eth.neu.edu:/discovery/home     155T  147T  9.0T  95% /home
vast1-mghpcc-eth.neu.edu:/discovery/scratch  2.2P  1.3P  867T  60% /scratch
vast1-mghpcc-eth.neu.edu:/datasets            40T   13T   28T  31% /datasets
vast1-mghpcc-eth.neu.edu:/work_project       3.7P  2.9P  795T  79% /projects
vast1-mghpcc-eth.neu.edu:/courses             36T   12T   25T  32% /courses
tmpfs                                         26G  4.0K   26G   1% /run/user/1825738601
tmpfs                                         26G  4.0K   26G   1% /run/user/1825771999
tmpfs                                         26G  4.0K   26G   1% /run/user/1825636447
tmpfs                                         26G  4.0K   26G   1% /run/user/101480
tmpfs                                         26G  4.0K   26G   1% /run/user/1825586042
tmpfs                                         26G  4.0K   26G   1% /run/user/101174
tmpfs                                         26G  4.0K   26G   1% /run/user/101507
tmpfs                                         26G  4.0K   26G   1% /run/user/1825810888
tmpfs                                         26G  4.0K   26G   1% /run/user/1825599022
tmpfs                                         26G  4.0K   26G   1% /run/user/1825585515
tmpfs                                         26G  4.0K   26G   1% /run/user/1825594703
tmpfs                                         26G   40K   26G   1% /run/user/92999
tmpfs                                         26G  4.0K   26G   1% /run/user/1825809851
tmpfs                                         26G  4.0K   26G   1% /run/user/20919
tmpfs                                         26G  4.0K   26G   1% /run/user/90984
tmpfs                                         26G  4.0K   26G   1% /run/user/21080
tmpfs                                         26G  4.0K   26G   1% /run/user/100843
tmpfs                                         26G  4.0K   26G   1% /run/user/100296
tmpfs                                         26G  4.0K   26G   1% /run/user/1825738566
tmpfs                                         26G  4.0K   26G   1% /run/user/93238
tmpfs                                         26G   16K   26G   1% /run/user/21055
tmpfs                                         26G  4.0K   26G   1% /run/user/101861
tmpfs                                         26G  4.0K   26G   1% /run/user/1825657717
tmpfs                                         26G  4.0K   26G   1% /run/user/101616
tmpfs                                         26G  4.0K   26G   1% /run/user/1825553069
tmpfs                                         26G  4.0K   26G   1% /run/user/21001
tmpfs                                         26G  4.0K   26G   1% /run/user/102043
tmpfs                                         26G  4.0K   26G   1% /run/user/100203
tmpfs                                         26G  4.0K   26G   1% /run/user/93471
tmpfs                                         26G  4.0K   26G   1% /run/user/101673
tmpfs                                         26G  4.0K   26G   1% /run/user/1825738504
tmpfs                                         26G  4.0K   26G   1% /run/user/1825842943
tmpfs                                         26G   12K   26G   1% /run/user/100625
tmpfs                                         26G  4.0K   26G   1% /run/user/90623
tmpfs                                         26G   28K   26G   1% /run/user/1825722135
tmpfs                                         26G  4.0K   26G   1% /run/user/1825757550
tmpfs                                         26G   28K   26G   1% /run/user/1825658587
tmpfs                                         26G  4.0K   26G   1% /run/user/90427
tmpfs                                         26G  4.0K   26G   1% /run/user/94306
tmpfs                                         26G  4.0K   26G   1% /run/user/94022
tmpfs                                         26G  4.0K   26G   1% /run/user/20174
tmpfs                                         26G  4.0K   26G   1% /run/user/100253
tmpfs                                         26G  4.0K   26G   1% /run/user/92338
tmpfs                                         26G  4.0K   26G   1% /run/user/93179
tmpfs                                         26G  4.0K   26G   1% /run/user/101545
tmpfs                                         26G  4.0K   26G   1% /run/user/100481
tmpfs                                         26G  4.0K   26G   1% /run/user/102066
tmpfs                                         26G  4.0K   26G   1% /run/user/90720
tmpfs                                         26G  8.0K   26G   1% /run/user/102182
tmpfs                                         26G  4.0K   26G   1% /run/user/1825856474
tmpfs                                         26G  4.0K   26G   1% /run/user/102057
tmpfs                                         26G  4.0K   26G   1% /run/user/102029
tmpfs                                         26G   28K   26G   1% /run/user/102052
tmpfs                                         26G  4.0K   26G   1% /run/user/1825736823
tmpfs                                         26G  4.0K   26G   1% /run/user/20450
tmpfs                                         26G  4.0K   26G   1% /run/user/1825729281
tmpfs                                         26G  4.0K   26G   1% /run/user/1825759859
tmpfs                                         26G   28K   26G   1% /run/user/102185
tmpfs                                         26G   28K   26G   1% /run/user/91534
tmpfs                                         26G  4.0K   26G   1% /run/user/101000
tmpfs                                         26G   72K   26G   1% /run/user/101623
tmpfs                                         26G  4.0K   26G   1% /run/user/1825784944
tmpfs                                         26G  4.0K   26G   1% /run/user/102238
tmpfs                                         26G  4.0K   26G   1% /run/user/92164
tmpfs                                         26G   28K   26G   1% /run/user/1825762050
tmpfs                                         26G  4.0K   26G   1% /run/user/102373
tmpfs                                         26G  4.0K   26G   1% /run/user/102305
tmpfs                                         26G  4.0K   26G   1% /run/user/101982
tmpfs                                         26G  4.0K   26G   1% /run/user/102479
tmpfs                                         26G  4.0K   26G   1% /run/user/102478
tmpfs                                         26G  4.0K   26G   1% /run/user/93100
tmpfs                                         26G  4.0K   26G   1% /run/user/92063
tmpfs                                         26G  4.0K   26G   1% /run/user/101975
tmpfs                                         26G  4.0K   26G   1% /run/user/1825808862
tmpfs                                         26G  4.0K   26G   1% /run/user/90532
tmpfs                                         26G   16K   26G   1% /run/user/1825738595
tmpfs                                         26G  4.0K   26G   1% /run/user/1825847196
tmpfs                                         26G  4.0K   26G   1% /run/user/92809
tmpfs                                         26G  4.0K   26G   1% /run/user/102156
tmpfs                                         26G  4.0K   26G   1% /run/user/102482
tmpfs                                         26G  4.0K   26G   1% /run/user/93532
tmpfs                                         26G  4.0K   26G   1% /run/user/91703
tmpfs                                         26G  4.0K   26G   1% /run/user/20441
tmpfs                                         26G  4.0K   26G   1% /run/user/101621
tmpfs                                         26G  4.0K   26G   1% /run/user/101954
tmpfs                                         26G  4.0K   26G   1% /run/user/90819
tmpfs                                         26G   16K   26G   1% /run/user/102849
tmpfs                                         26G  4.0K   26G   1% /run/user/102711
tmpfs                                         26G  4.0K   26G   1% /run/user/101211
tmpfs                                         26G  4.0K   26G   1% /run/user/93751
tmpfs                                         26G  4.0K   26G   1% /run/user/1825738520
tmpfs                                         26G  4.0K   26G   1% /run/user/101278
tmpfs                                         26G  4.0K   26G   1% /run/user/101664
tmpfs                                         26G  4.0K   26G   1% /run/user/1825555336
tmpfs                                         26G  4.0K   26G   1% /run/user/100746
tmpfs                                         26G  4.0K   26G   1% /run/user/102413
tmpfs                                         26G   16K   26G   1% /run/user/1825600606
tmpfs                                         26G  4.0K   26G   1% /run/user/1825642177
tmpfs                                         26G  4.0K   26G   1% /run/user/1825636000
tmpfs                                         26G  4.0K   26G   1% /run/user/50763
tmpfs                                         26G  4.0K   26G   1% /run/user/50202
tmpfs                                         26G  4.0K   26G   1% /run/user/102785
tmpfs                                         26G  4.0K   26G   1% /run/user/93023
tmpfs                                         26G  4.0K   26G   1% /run/user/100604
tmpfs                                         26G  4.0K   26G   1% /run/user/1825637287
tmpfs                                         26G  4.0K   26G   1% /run/user/102728
tmpfs                                         26G  4.0K   26G   1% /run/user/93750
tmpfs                                         26G  4.0K   26G   1% /run/user/1825811166
tmpfs                                         26G  4.0K   26G   1% /run/user/101247
tmpfs                                         26G  4.0K   26G   1% /run/user/1825738591
tmpfs                                         26G  4.0K   26G   1% /run/user/100839
tmpfs                                         26G  4.0K   26G   1% /run/user/102378
tmpfs                                         26G  4.0K   26G   1% /run/user/93117
tmpfs                                         26G  4.0K   26G   1% /run/user/91823
tmpfs                                         26G  4.0K   26G   1% /run/user/90909
tmpfs                                         26G  4.0K   26G   1% /run/user/1825720682
tmpfs                                         26G  4.0K   26G   1% /run/user/1825738602
tmpfs                                         26G  4.0K   26G   1% /run/user/1825586156
tmpfs                                         26G  4.0K   26G   1% /run/user/100806
tmpfs                                         26G  4.0K   26G   1% /run/user/103389
tmpfs                                         26G  4.0K   26G   1% /run/user/93492
tmpfs                                         26G  4.0K   26G   1% /run/user/20121
tmpfs                                         26G  4.0K   26G   1% /run/user/94266
tmpfs                                         26G  4.0K   26G   1% /run/user/92924
tmpfs                                         26G  4.0K   26G   1% /run/user/101062
tmpfs                                         26G  8.0K   26G   1% /run/user/102930
tmpfs                                         26G  4.0K   26G   1% /run/user/20044
tmpfs                                         26G  4.0K   26G   1% /run/user/1825829504
tmpfs                                         26G  4.0K   26G   1% /run/user/1825752584
tmpfs                                         26G  4.0K   26G   1% /run/user/93625
tmpfs                                         26G  4.0K   26G   1% /run/user/103413
tmpfs                                         26G  8.0K   26G   1% /run/user/1825816652
tmpfs                                         26G  4.0K   26G   1% /run/user/103640
tmpfs                                         26G  4.0K   26G   1% /run/user/91851
tmpfs                                         26G  4.0K   26G   1% /run/user/20987
tmpfs                                         26G  4.0K   26G   1% /run/user/101672
tmpfs                                         26G  4.0K   26G   1% /run/user/1825658901
tmpfs                                         26G  4.0K   26G   1% /run/user/101973
tmpfs                                         26G  4.0K   26G   1% /run/user/1825641682
tmpfs                                         26G  4.0K   26G   1% /run/user/90799
tmpfs                                         26G  4.0K   26G   1% /run/user/90340
tmpfs                                         26G  4.0K   26G   1% /run/user/101180
tmpfs                                         26G  4.0K   26G   1% /run/user/1825752482
tmpfs                                         26G  4.0K   26G   1% /run/user/91351
tmpfs                                         26G  4.0K   26G   1% /run/user/1825452260
tmpfs                                         26G  4.0K   26G   1% /run/user/102852
tmpfs                                         26G  4.0K   26G   1% /run/user/1825718213
tmpfs                                         26G  4.0K   26G   1% /run/user/50631
tmpfs                                         26G  4.0K   26G   1% /run/user/100619
tmpfs                                         26G   28K   26G   1% /run/user/93587
tmpfs                                         26G  4.0K   26G   1% /run/user/92980
tmpfs                                         26G  4.0K   26G   1% /run/user/1825846697
tmpfs                                         26G  4.0K   26G   1% /run/user/20349
tmpfs                                         26G  4.0K   26G   1% /run/user/1825555367
tmpfs                                         26G  4.0K   26G   1% /run/user/1825608774
tmpfs                                         26G  4.0K   26G   1% /run/user/1825809704
tmpfs                                         26G  4.0K   26G   1% /run/user/20506
tmpfs                                         26G  4.0K   26G   1% /run/user/93146
tmpfs                                         26G  4.0K   26G   1% /run/user/93284
tmpfs                                         26G  4.0K   26G   1% /run/user/1825755091
tmpfs                                         26G  4.0K   26G   1% /run/user/20065
tmpfs                                         26G  4.0K   26G   1% /run/user/101462
tmpfs                                         26G  4.0K   26G   1% /run/user/50932
tmpfs                                         26G  4.0K   26G   1% /run/user/20943
tmpfs                                         26G  4.0K   26G   1% /run/user/93789
tmpfs                                         26G  4.0K   26G   1% /run/user/20091
tmpfs                                         26G  4.0K   26G   1% /run/user/1825846674
tmpfs                                         26G  4.0K   26G   1% /run/user/93528
tmpfs                                         26G  4.0K   26G   1% /run/user/101713
tmpfs                                         26G  4.0K   26G   1% /run/user/1825758155
tmpfs                                         26G  4.0K   26G   1% /run/user/1825808932
tmpfs                                         26G  4.0K   26G   1% /run/user/93164
tmpfs                                         26G  4.0K   26G   1% /run/user/93942
tmpfs                                         26G  4.0K   26G   1% /run/user/92103
tmpfs                                         26G  4.0K   26G   1% /run/user/1825652827
tmpfs                                         26G  4.0K   26G   1% /run/user/91679
tmpfs                                         26G  4.0K   26G   1% /run/user/91005
tmpfs                                         26G  4.0K   26G   1% /run/user/1825891049
tmpfs                                         26G  4.0K   26G   1% /run/user/94301
tmpfs                                         26G  4.0K   26G   1% /run/user/92979
tmpfs                                         26G  4.0K   26G   1% /run/user/103464
tmpfs                                         26G  4.0K   26G   1% /run/user/20462
tmpfs                                         26G  4.0K   26G   1% /run/user/103676
tmpfs                                         26G  4.0K   26G   1% /run/user/92774
tmpfs                                         26G  4.0K   26G   1% /run/user/93939
tmpfs                                         26G  4.0K   26G   1% /run/user/101615
tmpfs                                         26G  4.0K   26G   1% /run/user/1825553943
tmpfs                                         26G  4.0K   26G   1% /run/user/1825519245
tmpfs                                         26G  4.0K   26G   1% /run/user/103351
tmpfs                                         26G  4.0K   26G   1% /run/user/20164
tmpfs                                         26G  4.0K   26G   1% /run/user/1825609892
tmpfs                                         26G  4.0K   26G   1% /run/user/103498
tmpfs                                         26G  4.0K   26G   1% /run/user/102427
tmpfs                                         26G  4.0K   26G   1% /run/user/103084
tmpfs                                         26G  4.0K   26G   1% /run/user/101087
tmpfs                                         26G  4.0K   26G   1% /run/user/92413
tmpfs                                         26G  4.0K   26G   1% /run/user/103694
tmpfs                                         26G  4.0K   26G   1% /run/user/100719
NAME    MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
sda       8:0    0 447.1G  0 disk 
├─sda1    8:1    0     1G  0 part /boot/efi
├─sda2    8:2    0     2G  0 part /boot
├─sda3    8:3    0  29.3G  0 part /var/lib/containers/storage/overlay
│                                 /var
├─sda4    8:4    0  19.5G  0 part /
├─sda5    8:5    0  19.5G  0 part [SWAP]
├─sda6    8:6    0   9.8G  0 part /srv
├─sda7    8:7    0   9.8G  0 part /var/log
├─sda8    8:8    0   9.8G  0 part /var/log/audit
├─sda9    8:9    0   9.8G  0 part /var/tmp
├─sda10   8:10   0     1M  0 part 
└─sda11   8:11   0 336.6G  0 part /tmp
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
2: eno8303: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc mq state DOWN group default qlen 1000
    link/ether a8:3c:a5:45:05:6a brd ff:ff:ff:ff:ff:ff
    altname enp4s0f0
3: eno8403: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc mq state DOWN group default qlen 1000
    link/ether a8:3c:a5:45:05:6b brd ff:ff:ff:ff:ff:ff
    altname enp4s0f1
4: internal: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000 qdisc mq state UP group default qlen 1000
    link/ether 6c:92:cf:1f:1c:e0 brd ff:ff:ff:ff:ff:ff
    altname enp49s0f0np0
    altname eno12399np0
    inet 10.99.200.107/16 brd 10.99.255.255 scope global noprefixroute internal
       valid_lft forever preferred_lft forever
5: eno12409np1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 6c:92:cf:1f:1c:e1 brd ff:ff:ff:ff:ff:ff
    altname enp49s0f1np1
    inet 129.10.0.146/24 scope global eno12409np1
       valid_lft forever preferred_lft forever
6: ens7f0np0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc mq state DOWN group default qlen 1000
    link/ether 04:32:01:3a:dc:d0 brd ff:ff:ff:ff:ff:ff
    altname enp202s0f0np0
7: ens7f1np1: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc mq state DOWN group default qlen 1000
    link/ether 04:32:01:3a:dc:d1 brd ff:ff:ff:ff:ff:ff
    altname enp202s0f1np1
8: ibp23s0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 4092 qdisc mq state DOWN group default qlen 256
    link/infiniband 00:00:04:4d:fe:80:00:00:00:00:00:00:58:a2:e1:03:00:b8:bf:3a brd 00:ff:ff:ff:ff:12:40:1b:ff:ff:00:00:00:00:00:00:ff:ff:ff:ff
NAME="Rocky Linux"
VERSION="9.3 (Blue Onyx)"
ID="rocky"
ID_LIKE="rhel centos fedora"
VERSION_ID="9.3"
PLATFORM_ID="platform:el9"
PRETTY_NAME="Rocky Linux 9.3 (Blue Onyx)"
ANSI_COLOR="0;32"
LOGO="fedora-logo-icon"
CPE_NAME="cpe:/o:rocky:rocky:9::baseos"
HOME_URL="https://rockylinux.org/"
BUG_REPORT_URL="https://bugs.rockylinux.org/"
SUPPORT_END="2032-05-31"
ROCKY_SUPPORT_PRODUCT="Rocky-Linux-9"
ROCKY_SUPPORT_PRODUCT_VERSION="9.3"
REDHAT_SUPPORT_PRODUCT="Rocky Linux"
REDHAT_SUPPORT_PRODUCT_VERSION="9.3"