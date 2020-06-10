import numpy as np

hk = np.array([
  [-0.3239392,-0.11337998,-0.032942563,0.07580175,0.45070288,0.2650364,0.11153643,0.024855405,-0.07285671,-0.32288033,-0.060177375,-0.02842745,-0.07222776,0.00055096333,0.1868228,0.11146693,0.064815864,0.036820166,0.018493796,-0.21274535,0.29683858,0.12265697,-0.026541542,-0.17767394,-0.24255615,-0.0068985866,-0.012452329,0.024672164,0.050175518,-0.041905433,0.31273997,0.12985037,-0.006229551,-0.1494478,-0.3329635,-0.061842434,-0.0118167065,0.07374169,0.08977938,0.037367523,0.1135919,0.060771506,0.008980605,-0.07774685,-0.17594422,0.06978432,0.071010865,0.06276852,0.030209975,-0.043314237,0.0024544026,-0.07467184,-0.035530545,-0.034002908,-0.030829871,0.10429189,0.08737859,0.032715444,-0.024612552,-0.119004324,0.062375184,-0.0073233936,0.007884055,-0.019955138,-0.03357349,0.13285053,0.11589758,0.05951661,-0.0075417324,-0.11416993,0.09652612,0.025420692,0.014884248,-0.049139194,-0.13444619,0.19471525,0.16924483,0.06639172,-0.035236623,-0.17367141,0.18136448,0.05175171,0.020663224,-0.071326524,-0.13835846,0.1807895,0.118350685,0.047089968,-0.030362356,-0.16823848,0.14778261,0.026077768,0.013680502,-0.017440986,-0.13289718],
  [0.1420734,0.08510852,0.06846222,0.15625711,0.25373653,0.18352175,0.109058976,0.06510974,0.01721669,0.031364422,0.060636126,0.062080193,0.0076769246,0.05463681,-0.002968216,0.085447274,0.03326558,0.051542703,-0.009436845,-0.023591911,0.035232678,0.072640255,0.035294864,0.10257799,0.076719485,-0.042231083,-0.055907972,-0.021073882,-0.019928271,-0.05065051,0.02898552,0.031704646,0.02237888,0.105203435,0.07595296,0.0075309286,-0.009210716,0.06114157,0.01853814,0.037619263,0.05699757,0.09551766,0.069477595,0.1381958,0.139975,0.12028402,0.05050851,0.10986302,0.124023974,0.090451725,0.09432243,0.091852404,0.027571892,0.1270787,0.15754494,0.09118293,-0.019948931,0.014654861,0.04705165,0.037819758,0.120953545,0.15070906,0.10479698,0.1406331,0.13986874,0.017607315,-0.07398291,0.022469323,0.040225364,-0.008949921,0.20682491,0.23218809,0.10815249,0.22280824,0.22019078,0.051640753,0.02069761,0.04371747,0.13135162,0.12013734,0.26092404,0.22851557,0.20840089,0.2736868,0.26873574,0.09926658,0.10758164,0.099638574,0.20441334,0.19017218,0.17152348,0.10475303,0.1475557,0.19604173,0.1882001],
  [0.04018861,-0.017996583,0.023029502,-0.005051506,-0.022971475,-0.1261477,-0.020380683,0.04146348,0.077492,0.12822239,-0.129408,0.0005932614,0.14150047,0.17219624,0.17371997,-0.06907773,0.013828622,0.070308246,0.15681423,0.19421376,-0.21334927,-0.16796033,-0.103040114,-0.020450555,0.1326401,-0.1480538,-0.11229291,-0.15610904,-0.12632973,-0.08438683,-0.18180935,-0.14803654,-0.049891707,0.034686144,0.16701798,0.12195338,0.17283376,0.15730211,0.14741264,0.11986949,-0.05807737,-0.06329175,-0.04060127,-0.005817694,0.012247857,0.106478766,0.09629202,0.070049256,0.041264024,0.048587214,-0.0725077,-0.15216003,-0.18595284,-0.17563103,-0.16735671,0.087994836,0.13201788,0.06220985,0.030612968,0.113183506,0.009526249,0.0591604,0.07103614,0.04526949,0.04290645,0.05574472,0.09772989,0.07961769,0.050270125,0.11020743,-0.022763338,0.03330987,0.010444947,-0.023944046,-0.070581004,0.0015914532,0.10018055,0.08153844,0.086600654,0.15582569,-0.01271664,0.0033375432,0.023895908,-0.023109637,-0.044151176,-0.02091326,0.026877573,0.03912136,-0.0062881103,0.04905418,0.023628935,0.046474755,0.0366221,-0.012842071,-0.044407137],
  [0.13116798,0.078877345,0.08618855,0.0397589,-0.105776265,0.062194422,0.038235396,0.024085455,-0.068198614,-0.062084578,0.08961247,0.054975454,-0.014667932,-0.0898122,-0.19056663,-0.0317014,0.021666782,-0.004738606,-0.033009816,0.05876011,-0.07878128,-0.031323064,0.067245334,0.08933557,0.09653397,0.109447524,0.17888537,0.14017844,0.12757589,0.110307366,-0.16309346,-0.12512173,-0.07360147,-0.11474082,-0.07564989,-0.024630893,-0.06840722,-0.076514475,0.0035568448,0.004056047,-0.07986678,-0.06448548,0.0015890796,-0.032684237,-0.022829132,0.10057017,0.13788027,0.14930668,0.19254899,0.19544093,-0.0040404503,-0.0053235516,-0.027648656,-0.05551999,-0.07731741,-0.03529336,-0.13316576,-0.13372529,-0.05489515,0.0028558709,0.15912506,0.12219316,0.10451392,0.035550106,0.021741075,0.050988834,0.0072100367,0.0052243746,0.07443768,0.09850988,0.02850966,0.060196266,-0.015666345,0.0010376528,-0.03408707,-0.06473634,-0.08175681,-0.14601365,-0.02453482,0.022985328,0.15572132,0.092229225,0.09946154,0.11357728,0.1271754,-0.09603449,-0.034966763,-0.08893859,0.02047761,0.087115586,0.10656323,0.047030777,0.112212606,0.09603554,0.08933289],
  [-0.088171564,0.009865733,-0.056503452,-0.049548924,-0.091643766,-0.07533288,-0.059531264,-0.06286083,-0.078442775,-0.123059735,-0.03897537,-0.067628436,-0.070352994,-0.06695945,-0.023382561,0.019271566,0.05480717,0.012421372,0.011589021,-0.017817207,-0.042906247,0.002355028,0.02319726,-0.01608768,0.0011625539,-0.030430235,-0.04444383,-0.025680432,-0.010333815,-0.011405606,0.041079573,0.063638754,0.08079727,0.05821741,0.043380275,0.121464975,0.17056866,0.19689658,0.16689618,0.19669646,0.19354296,0.18543057,0.22203404,0.17840306,0.3036915,0.21780993,0.2593733,0.23224865,0.2567961,0.31307065,0.14226441,0.18894207,0.15840183,0.15271868,0.21331929,0.09070814,0.13212875,0.059991352,0.052131504,0.07416575,0.056056976,0.012913315,0.03892665,0.07559712,-0.0054677147,0.096212216,0.13380511,0.03575134,0.016096078,0.06328053,0.054980624,0.056749392,0.061467372,0.053212408,0.047080185,0.19206513,0.28326923,0.18197304,0.14275838,0.18768777,0.097973384,0.0709609,0.08557844,0.082053624,0.02616202,0.06926366,0.15192097,0.11259027,0.087889835,0.061436933,0.04282889,-0.0039229984,0.007513444,-0.009571113,0.020211825],
  [-0.0037760779,-0.13179146,0.07096667,-0.043755855,-0.011226945,-0.094460726,0.0054000854,0.0019995715,-0.014328291,-0.18493527,-0.21051103,-0.055510897,-0.014789193,-0.12153784,-0.1223602,-0.1761947,-0.021356698,0.005530652,-0.053533,-0.10781519,-0.1818207,-0.14492236,-0.06318554,-0.15928236,-0.1383115,-0.20702082,-0.11472172,-0.0042514317,-0.12257238,-0.20374708,-0.17465432,-0.15455824,-0.09454618,-0.21257113,-0.11559865,-0.15600315,-0.18662983,0.026817992,-0.17020395,-0.18986963,-0.15352522,-0.22295249,-0.13087097,-0.20507716,-0.2173632,-0.07079377,-0.14379048,0.008239062,-0.07582122,-0.16545212,-0.20659205,-0.18420662,-0.15964438,-0.16945334,-0.21743278,-0.010375482,-0.007900523,0.04335807,-0.06483819,-0.013050194,0.013544674,0.036887024,0.02198425,-0.0599944,0.04247541,0.17628038,0.07812442,0.17129816,0.103727974,0.15420341,0.03672198,0.08312895,0.027311077,-0.03981083,-0.024819389,0.16640401,0.09216863,0.19690342,0.14023632,0.10872054,0.13175546,0.06531022,0.10417272,-0.00088119804,0.048181456,0.12189638,0.04465506,0.13432552,0.05756966,0.053898122,0.103183605,0.043991458,0.06547835,-0.015299992,-0.007346281],
  [0.21653774,0.013263109,-0.078717045,-0.06726644,-0.024988873,0.304764,0.15437204,0.060651474,-0.058377933,-0.17686866,0.45074248,0.113033935,-0.15980156,-0.22855948,-0.29468253,0.30527923,0.14988013,0.035719797,-0.08647218,-0.28677768,0.019764546,0.09154312,0.091132656,0.10095248,0.12274905,-0.1663244,-0.17098244,-0.14113814,-0.023769585,-0.080204606,0.09224401,0.059069347,0.1341813,0.13161226,0.14854796,-0.19662142,-0.03760658,0.051491037,0.11530813,-0.04651712,-0.08065304,0.05111749,0.08597893,0.14627443,0.19931985,-0.06183107,-0.055830825,-0.010132727,0.058949035,0.07689935,-0.07982099,-0.097157404,-0.08968834,-0.0018619456,0.072180755,-0.049898602,-0.035864227,-0.018923555,0.07593354,0.094832145,0.0549807,0.0039770105,0.009035099,0.09442962,-0.02328432,-0.046941202,-0.06152642,0.030471971,0.027980912,0.098124534,-0.13367553,-0.04892136,-0.03395856,0.16035853,0.03745153,-0.09640249,0.052515227,0.17785934,0.14487742,0.22724764,-0.14360295,-0.05603592,0.10640992,0.192392,0.10722492,-0.17914537,0.03292482,0.14110199,0.27989587,0.21889125,-0.12085278,-0.090931684,0.052686743,0.082139164,0.13683982],
  [-0.0843019,0.00985224,0.023621446,0.024931302,-0.027073743,-0.009235369,0.0034471164,-0.029438803,-0.07709291,-0.17870495,0.0021193507,0.002134587,0.0071895164,-0.038739603,-0.08308519,0.043856177,0.045879442,0.026029823,0.019998237,-0.05013339,0.16283189,0.118561976,0.10345054,0.065555535,0.12214767,0.058727954,0.09273706,0.026574217,0.08427032,0.05826178,0.21061398,0.1367209,0.14450309,0.09384106,0.14875458,-0.12999453,-0.033357225,-0.035571486,-0.009834668,-0.104948856,0.07617224,0.042980526,0.03802828,-0.00283011,0.022097953,-0.074085504,0.016929355,-0.03790694,-0.016656045,-0.090660855,0.080607675,0.02048119,-0.003689495,-0.01988916,0.015500994,-0.0899113,-0.039003436,-0.06448704,-0.08770015,-0.11026822,0.118583836,0.08075629,0.07872363,0.06263568,0.031481843,-0.13163958,-0.054131176,-0.06391382,-0.10326698,-0.137821,0.12411331,0.08887158,0.08904498,0.075617634,0.0064224745,-0.2647559,-0.14883514,-0.17475241,-0.20723367,-0.29483017,0.075915635,0.058923148,0.04348555,0.0389697,0.0016421829,-0.20066804,-0.085368775,-0.13669376,-0.14635469,-0.17593716,0.07443821,0.041885063,0.07376814,0.035028562,0.0147961825]
])

hb = np.array([
  [0.0037867997],
  [-1.228012],
  [0.62966007],
  [1.030773e-05],
  [1.319621],
  [1.2756099],
  [-0.27516374],
  [0.40542313]
])

ok = np.array([
  [0.3430452,0.60147715,0.83039737,-0.19238268,0.9581698,0.4103913,0.044885825,0.20480901],
  [0.4371591,0.41951537,0.92788106,-0.07608234,0.433688,0.2345186,0.15374932,0.05827792],
  [-0.1709166,-0.063065015,-0.049677588,-0.70311546,0.44368613,-0.42255813,-0.40082666,-0.38919887]
])

ob = np.array([
  [0.3892914],
  [1.4834269],
  [-1.8899791]
])

def classify(x):
  return np.squeeze(np.argmax(np.add(np.matmul(np.maximum(np.add(np.matmul(x, hk.T), hb.T), 0), ok.T), ob), axis=1))

def tf_classify(x):
  return classify(x)