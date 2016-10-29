```sh
$ ./get-data.sh
```

```r
$ R
> install.packages("kernlab", quiet=TRUE)
Warning in install.packages("kernlab", quiet = TRUE) :
  'lib = "/usr/local/lib/R/site-library"' is not writable
Would you like to use a personal library instead?  (y/n) y
Would you like to create a personal library
~/R/x86_64-pc-linux-gnu-library/3.2
to install packages into?  (y/n) y
--- Please select a CRAN mirror for use in this session ---
HTTPS CRAN mirror

 1: 0-Cloud [https]                2: Austria [https]
 3: Chile [https]                  4: China (Beijing 4) [https]
 5: Colombia (Cali) [https]        6: France (Lyon 2) [https]
 7: Germany (Münster) [https]      8: Iceland [https]
 9: Mexico (Mexico City) [https]  10: Russia (Moscow) [https]
11: Spain (A Coruña) [https]      12: Switzerland [https]
13: UK (Bristol) [https]          14: UK (Cambridge) [https]
15: USA (CA 1) [https]            16: USA (KS) [https]
17: USA (MI 1) [https]            18: USA (TN) [https]
19: USA (TX) [https]              20: USA (WA) [https]
21: (HTTP mirrors)

Selection: 21
HTTP CRAN mirror

  1: 0-Cloud                        2: Algeria
  3: Argentina (La Plata)           4: Australia (Canberra)
  5: Australia (Melbourne)          6: Austria
  7: Belgium (Antwerp)              8: Belgium (Ghent)
  9: Brazil (BA)                   10: Brazil (PR)
 11: Brazil (RJ)                   12: Brazil (SP 1)
 13: Brazil (SP 2)                 14: Canada (BC)
 15: Canada (NS)                   16: Canada (ON)
 17: Chile                         18: China (Beijing 2)
 19: China (Beijing 3)             20: China (Beijing 4)
 21: China (Xiamen)                22: Colombia (Cali)
 23: Czech Republic                24: Denmark
 25: Ecuador                       26: El Salvador
 27: Estonia                       28: France (Lyon 1)
 29: France (Lyon 2)               30: France (Montpellier)
 31: France (Paris 1)              32: France (Paris 2)
 33: Germany (Berlin)              34: Germany (Göttingen)
 35: Germany (Münster)             36: Greece
 37: Hungary                       38: Iceland
 39: India                         40: Indonesia (Jakarta)
 41: Iran                          42: Ireland
 43: Italy (Milano)                44: Italy (Padua) [https]
 45: Italy (Padua)                 46: Italy (Palermo)
 47: Japan (Tokyo)                 48: Japan (Yamagata)
 49: Korea (Seoul 1)               50: Korea (Seoul 2)
 51: Korea (Ulsan)                 52: Lebanon
 53: Mexico (Mexico City)          54: Mexico (Texcoco)
 55: Netherlands (Amsterdam)       56: Netherlands (Utrecht)
 57: New Zealand                   58: Norway
 59: Philippines                   60: Poland
 61: Portugal (Lisbon)             62: Russia (Moscow)
 63: Slovakia                      64: South Africa (Cape Town)
 65: South Africa (Johannesburg)   66: Spain (A Coruña)
 67: Spain (Madrid)                68: Sweden
 69: Switzerland                   70: Taiwan (Chungli)
 71: Taiwan (Taipei)               72: Thailand
 73: Turkey (Denizli)              74: Turkey (Mersin)
 75: UK (Bristol)                  76: UK (Cambridge)
 77: UK (London 1)                 78: UK (London 2)
 79: UK (St Andrews)               80: USA (CA 1)
 81: USA (CA 2)                    82: USA (CO)
 83: USA (IA)                      84: USA (IN)
 85: USA (KS)                      86: USA (MD)
 87: USA (MI 1)                    88: USA (MI 2)
 89: USA (MO)                      90: USA (NC)
 91: USA (OH 1)                    92: USA (OH 2)
 93: USA (OR)                      94: USA (PA 1)
 95: USA (PA 2)                    96: USA (TN)
 97: USA (TX)                      98: USA (WA)
 99: Venezuela                    100: Vietnam


Selection: 47
> install.packages("e1071", quiet=TRUE)
> install.packages("randomForest", quiet=TRUE)
> install.packages("caret", quiet=TRUE)
also installing the dependencies ‘colorspace’, ‘minqa’, ‘nloptr’, ‘RcppEigen’, ‘RColorBrewer’, ‘dichromat’, ‘munsell’, ‘labeling’, ‘lme4’, ‘SparseM’, ‘MatrixModels’, ‘stringi’, ‘magrittr’, ‘digest’, ‘gtable’, ‘scales’, ‘proto’, ‘pbkrtest’, ‘quantreg’, ‘iterators’, ‘Rcpp’, ‘stringr’, ‘ggplot2’, ‘car’, ‘foreach’, ‘plyr’, ‘reshape2’

> q()
Save workspace image? [y/n/c]: y
```

```r
$ cat [2-4].R | R --save
pred     no   yes
  no  11682  1041
  yes   302   539
> (prec <- conf.mat["yes", "yes"]/sum(conf.mat["yes", ])) # 適合率
[1] 0.6409037
> (rec <- conf.mat["yes", "yes"]/sum(conf.mat[, "yes"])) # 再現率
[1] 0.3411392
> (f.value <- 2 * prec * rec/(prec + rec)) # F-値
[1] 0.4452705
> (acc <- sum(diag(conf.mat))/sum(conf.mat)) # 正解率
[1] 0.9009879

```

```r
$ cat 5.R | R --save
pred     no   yes
  no  11503   804
  yes   481   776
> (prec <- conf.mat["yes", "yes"]/sum(conf.mat["yes", ])) # 適合率
[1] 0.6173429
> (rec <- conf.mat["yes", "yes"]/sum(conf.mat[, "yes"])) # 再現率
[1] 0.4911392
> (f.value <- 2 * prec * rec/(prec + rec)) # F-値
[1] 0.5470568
> (acc <- sum(diag(conf.mat))/sum(conf.mat)) # 正解率
[1] 0.9052639
```
