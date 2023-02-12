# mini-sms-classify
>垃圾邮件分类（朴素贝叶斯算法），使用flask+vue

### 背景
* 2002年，Paul Graham提出使用"贝叶斯推断"过滤垃圾邮件。他说，这样做的效果，好得不可思议。1000封垃圾邮件可以过滤掉995封，且没有一个误判。
* 因为典型的垃圾邮件词汇在垃圾邮件中会以更高的频率出现，所以在做贝叶斯公式计算时，肯定会被识别出来。之后用最高频的15个垃圾词汇做联合概率计算，联合概率的结果超过90%将说明它是垃圾邮件。
* 用贝叶斯过滤器可以识别很多改写过的垃圾邮件，而且错判率非常低。甚至不要求对初始值有多么精确，精度会在随后计算中逐渐逼近真实情况。
* 贝叶斯公式其实类似人脑的学习过程，这使得他在机器学习中广泛的应用：baby学一个新单词，他一开始并不知道这个词是什么意思，但是他可以根据当时的情景，先来个猜测（先验概率/主观判断）。一有机会，他就会在不同的场合说出这个词，然后观察你的反应。如果我告诉他用对了，他就会进一步记住这个词的意思，如果我告诉他用错了，他就会进行相应调整。（可能性函数/调整因子）。经过这样反复的猜测、试探、调整主观判断，就是贝叶斯定理思维的过程。
* 这过程就是：主观判断（先验概率）+ 搜集的信息（可能性函数）= 优化概率（后验概率）
* <svg xmlns="http://www.w3.org/2000/svg" width="45.744ex" height="5.475ex" viewBox="0 -1460 20218.9 2420" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" style=""><defs><path id="MJX-174-TEX-I-1D443" d="M287 628Q287 635 230 637Q206 637 199 638T192 648Q192 649 194 659Q200 679 203 681T397 683Q587 682 600 680Q664 669 707 631T751 530Q751 453 685 389Q616 321 507 303Q500 302 402 301H307L277 182Q247 66 247 59Q247 55 248 54T255 50T272 48T305 46H336Q342 37 342 35Q342 19 335 5Q330 0 319 0Q316 0 282 1T182 2Q120 2 87 2T51 1Q33 1 33 11Q33 13 36 25Q40 41 44 43T67 46Q94 46 127 49Q141 52 146 61Q149 65 218 339T287 628ZM645 554Q645 567 643 575T634 597T609 619T560 635Q553 636 480 637Q463 637 445 637T416 636T404 636Q391 635 386 627Q384 621 367 550T332 412T314 344Q314 342 395 342H407H430Q542 342 590 392Q617 419 631 471T645 554Z"></path><path id="MJX-174-TEX-N-28" d="M94 250Q94 319 104 381T127 488T164 576T202 643T244 695T277 729T302 750H315H319Q333 750 333 741Q333 738 316 720T275 667T226 581T184 443T167 250T184 58T225 -81T274 -167T316 -220T333 -241Q333 -250 318 -250H315H302L274 -226Q180 -141 137 -14T94 250Z"></path><path id="MJX-174-TEX-I-1D446" d="M308 24Q367 24 416 76T466 197Q466 260 414 284Q308 311 278 321T236 341Q176 383 176 462Q176 523 208 573T273 648Q302 673 343 688T407 704H418H425Q521 704 564 640Q565 640 577 653T603 682T623 704Q624 704 627 704T632 705Q645 705 645 698T617 577T585 459T569 456Q549 456 549 465Q549 471 550 475Q550 478 551 494T553 520Q553 554 544 579T526 616T501 641Q465 662 419 662Q362 662 313 616T263 510Q263 480 278 458T319 427Q323 425 389 408T456 390Q490 379 522 342T554 242Q554 216 546 186Q541 164 528 137T492 78T426 18T332 -20Q320 -22 298 -22Q199 -22 144 33L134 44L106 13Q83 -14 78 -18T65 -22Q52 -22 52 -14Q52 -11 110 221Q112 227 130 227H143Q149 221 149 216Q149 214 148 207T144 186T142 153Q144 114 160 87T203 47T255 29T308 24Z"></path><path id="MJX-174-TEX-N-7C" d="M139 -249H137Q125 -249 119 -235V251L120 737Q130 750 139 750Q152 750 159 735V-235Q151 -249 141 -249H139Z"></path><path id="MJX-174-TEX-I-1D44A" d="M436 683Q450 683 486 682T553 680Q604 680 638 681T677 682Q695 682 695 674Q695 670 692 659Q687 641 683 639T661 637Q636 636 621 632T600 624T597 615Q597 603 613 377T629 138L631 141Q633 144 637 151T649 170T666 200T690 241T720 295T759 362Q863 546 877 572T892 604Q892 619 873 628T831 637Q817 637 817 647Q817 650 819 660Q823 676 825 679T839 682Q842 682 856 682T895 682T949 681Q1015 681 1034 683Q1048 683 1048 672Q1048 666 1045 655T1038 640T1028 637Q1006 637 988 631T958 617T939 600T927 584L923 578L754 282Q586 -14 585 -15Q579 -22 561 -22Q546 -22 542 -17Q539 -14 523 229T506 480L494 462Q472 425 366 239Q222 -13 220 -15T215 -19Q210 -22 197 -22Q178 -22 176 -15Q176 -12 154 304T131 622Q129 631 121 633T82 637H58Q51 644 51 648Q52 671 64 683H76Q118 680 176 680Q301 680 313 683H323Q329 677 329 674T327 656Q322 641 318 637H297Q236 634 232 620Q262 160 266 136L501 550L499 587Q496 629 489 632Q483 636 447 637Q428 637 422 639T416 648Q416 650 418 660Q419 664 420 669T421 676T424 680T428 682T436 683Z"></path><path id="MJX-174-TEX-N-29" d="M60 749L64 750Q69 750 74 750H86L114 726Q208 641 251 514T294 250Q294 182 284 119T261 12T224 -76T186 -143T145 -194T113 -227T90 -246Q87 -249 86 -250H74Q66 -250 63 -250T58 -247T55 -238Q56 -237 66 -225Q221 -64 221 250T66 725Q56 737 55 738Q55 746 60 749Z"></path><path id="MJX-174-TEX-N-3D" d="M56 347Q56 360 70 367H707Q722 359 722 347Q722 336 708 328L390 327H72Q56 332 56 347ZM56 153Q56 168 72 173H708Q722 163 722 153Q722 140 707 133H70Q56 140 56 153Z"></path><path id="MJX-174-TEX-N-2217" d="M229 286Q216 420 216 436Q216 454 240 464Q241 464 245 464T251 465Q263 464 273 456T283 436Q283 419 277 356T270 286L328 328Q384 369 389 372T399 375Q412 375 423 365T435 338Q435 325 425 315Q420 312 357 282T289 250L355 219L425 184Q434 175 434 161Q434 146 425 136T401 125Q393 125 383 131T328 171L270 213Q283 79 283 63Q283 53 276 44T250 35Q231 35 224 44T216 63Q216 80 222 143T229 213L171 171Q115 130 110 127Q106 124 100 124Q87 124 76 134T64 161Q64 166 64 169T67 175T72 181T81 188T94 195T113 204T138 215T170 230T210 250L74 315Q65 324 65 338Q65 353 74 363T98 374Q106 374 116 368T171 328L229 286Z"></path><path id="MJX-174-TEX-N-2B" d="M56 237T56 250T70 270H369V420L370 570Q380 583 389 583Q402 583 409 568V270H707Q722 262 722 250T707 230H409V-68Q401 -82 391 -82H389H387Q375 -82 369 -68V230H70Q56 237 56 250Z"></path><path id="MJX-174-TEX-I-1D43B" d="M228 637Q194 637 192 641Q191 643 191 649Q191 673 202 682Q204 683 219 683Q260 681 355 681Q389 681 418 681T463 682T483 682Q499 682 499 672Q499 670 497 658Q492 641 487 638H485Q483 638 480 638T473 638T464 637T455 637Q416 636 405 634T387 623Q384 619 355 500Q348 474 340 442T328 395L324 380Q324 378 469 378H614L615 381Q615 384 646 504Q674 619 674 627T617 637Q594 637 587 639T580 648Q580 650 582 660Q586 677 588 679T604 682Q609 682 646 681T740 680Q802 680 835 681T871 682Q888 682 888 672Q888 645 876 638H874Q872 638 869 638T862 638T853 637T844 637Q805 636 794 634T776 623Q773 618 704 340T634 58Q634 51 638 51Q646 48 692 46H723Q729 38 729 37T726 19Q722 6 716 0H701Q664 2 567 2Q533 2 504 2T458 2T437 1Q420 1 420 10Q420 15 423 24Q428 43 433 45Q437 46 448 46H454Q481 46 514 49Q520 50 522 50T528 55T534 64T540 82T547 110T558 153Q565 181 569 198Q602 330 602 331T457 332H312L279 197Q245 63 245 58Q245 51 253 49T303 46H334Q340 38 340 37T337 19Q333 6 327 0H312Q275 2 178 2Q144 2 115 2T69 2T48 1Q31 1 31 10Q31 12 34 24Q39 43 44 45Q48 46 59 46H65Q92 46 125 49Q139 52 144 61Q147 65 216 339T285 628Q285 635 228 637Z"></path></defs><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><g data-mml-node="math"><g data-mml-node="mi"><use xlink:href="#MJX-174-TEX-I-1D443"></use></g><g data-mml-node="mo" transform="translate(751, 0)"><use xlink:href="#MJX-174-TEX-N-28"></use></g><g data-mml-node="mi" transform="translate(1140, 0)"><use xlink:href="#MJX-174-TEX-I-1D446"></use></g><g data-mml-node="TeXAtom" data-mjx-texclass="ORD" transform="translate(1785, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-174-TEX-N-7C"></use></g></g><g data-mml-node="mi" transform="translate(2063, 0)"><use xlink:href="#MJX-174-TEX-I-1D44A"></use></g><g data-mml-node="mo" transform="translate(3111, 0)"><use xlink:href="#MJX-174-TEX-N-29"></use></g><g data-mml-node="mo" transform="translate(3777.8, 0)"><use xlink:href="#MJX-174-TEX-N-3D"></use></g><g data-mml-node="mfrac" transform="translate(4833.6, 0)"><g data-mml-node="mrow" transform="translate(4383.4, 710)"><g data-mml-node="mi"><use xlink:href="#MJX-174-TEX-I-1D443"></use></g><g data-mml-node="mo" transform="translate(751, 0)"><use xlink:href="#MJX-174-TEX-N-28"></use></g><g data-mml-node="mi" transform="translate(1140, 0)"><use xlink:href="#MJX-174-TEX-I-1D44A"></use></g><g data-mml-node="TeXAtom" data-mjx-texclass="ORD" transform="translate(2188, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-174-TEX-N-7C"></use></g></g><g data-mml-node="mi" transform="translate(2466, 0)"><use xlink:href="#MJX-174-TEX-I-1D446"></use></g><g data-mml-node="mo" transform="translate(3111, 0)"><use xlink:href="#MJX-174-TEX-N-29"></use></g><g data-mml-node="mo" transform="translate(3722.2, 0)"><use xlink:href="#MJX-174-TEX-N-2217"></use></g><g data-mml-node="mi" transform="translate(4444.4, 0)"><use xlink:href="#MJX-174-TEX-I-1D443"></use></g><g data-mml-node="mo" transform="translate(5195.4, 0)"><use xlink:href="#MJX-174-TEX-N-28"></use></g><g data-mml-node="mi" transform="translate(5584.4, 0)"><use xlink:href="#MJX-174-TEX-I-1D446"></use></g><g data-mml-node="mo" transform="translate(6229.4, 0)"><use xlink:href="#MJX-174-TEX-N-29"></use></g></g><g data-mml-node="mrow" transform="translate(220, -710)"><g data-mml-node="mi"><use xlink:href="#MJX-174-TEX-I-1D443"></use></g><g data-mml-node="mo" transform="translate(751, 0)"><use xlink:href="#MJX-174-TEX-N-28"></use></g><g data-mml-node="mi" transform="translate(1140, 0)"><use xlink:href="#MJX-174-TEX-I-1D44A"></use></g><g data-mml-node="TeXAtom" data-mjx-texclass="ORD" transform="translate(2188, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-174-TEX-N-7C"></use></g></g><g data-mml-node="mi" transform="translate(2466, 0)"><use xlink:href="#MJX-174-TEX-I-1D446"></use></g><g data-mml-node="mo" transform="translate(3111, 0)"><use xlink:href="#MJX-174-TEX-N-29"></use></g><g data-mml-node="mo" transform="translate(3722.2, 0)"><use xlink:href="#MJX-174-TEX-N-2217"></use></g><g data-mml-node="mi" transform="translate(4444.4, 0)"><use xlink:href="#MJX-174-TEX-I-1D443"></use></g><g data-mml-node="mo" transform="translate(5195.4, 0)"><use xlink:href="#MJX-174-TEX-N-28"></use></g><g data-mml-node="mi" transform="translate(5584.4, 0)"><use xlink:href="#MJX-174-TEX-I-1D446"></use></g><g data-mml-node="mo" transform="translate(6229.4, 0)"><use xlink:href="#MJX-174-TEX-N-29"></use></g><g data-mml-node="mo" transform="translate(6840.7, 0)"><use xlink:href="#MJX-174-TEX-N-2B"></use></g><g data-mml-node="mi" transform="translate(7840.9, 0)"><use xlink:href="#MJX-174-TEX-I-1D443"></use></g><g data-mml-node="mo" transform="translate(8591.9, 0)"><use xlink:href="#MJX-174-TEX-N-28"></use></g><g data-mml-node="mi" transform="translate(8980.9, 0)"><use xlink:href="#MJX-174-TEX-I-1D44A"></use></g><g data-mml-node="TeXAtom" data-mjx-texclass="ORD" transform="translate(10028.9, 0)"><g data-mml-node="mo"><use xlink:href="#MJX-174-TEX-N-7C"></use></g></g><g data-mml-node="mi" transform="translate(10306.9, 0)"><use xlink:href="#MJX-174-TEX-I-1D43B"></use></g><g data-mml-node="mo" transform="translate(11194.9, 0)"><use xlink:href="#MJX-174-TEX-N-29"></use></g><g data-mml-node="mo" transform="translate(11806.1, 0)"><use xlink:href="#MJX-174-TEX-N-2217"></use></g><g data-mml-node="mi" transform="translate(12528.3, 0)"><use xlink:href="#MJX-174-TEX-I-1D443"></use></g><g data-mml-node="mo" transform="translate(13279.3, 0)"><use xlink:href="#MJX-174-TEX-N-28"></use></g><g data-mml-node="mi" transform="translate(13668.3, 0)"><use xlink:href="#MJX-174-TEX-I-1D43B"></use></g><g data-mml-node="mo" transform="translate(14556.3, 0)"><use xlink:href="#MJX-174-TEX-N-29"></use></g></g><rect width="15145.3" height="60" x="120" y="220"></rect></g></g></g></svg>
* 其中用W表示某个词，现在需要计算P(S|W)的值，即在某个词语（W）已经存在的条件下，垃圾邮件（S）的概率有多大。
## 朴素贝叶斯分类
## 如何运行
### server端
```bash
cd server
# 创建虚拟环境
python -m venv env
# 激活虚拟环境
source env/bin/activate
# 安装依赖包
pip install -r requirements.txt
#下载jieba，中文分词
https://pypi.org/project/jieba/#files
解压后，打开，使用
python setup.py install, 安装到特定的python运行环境
# 启动flask
python app.py

```
### client端
```bash
npm install
npm run serve
```
* 解释 ： models文件夹下面是英文分词的模型
* models2文件夹下面是中文分词的模型
* 其中的Analyzer.py是评估中文分词效果，运行时候需要将