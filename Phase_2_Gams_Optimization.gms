
sets
j index for jobs
d index for days
p index for persons /1*3/
m index for month /1*12/;


parameter C(j,d);
parameter R(m)/1 791578.9474,
               2 908461.5385,
               3 978181.8182,
               4 737600,
               5 887692.3077,
               6 1144545.455,
               7 1368461.538,
               8 1104347.826,
               9 1416666.667,
               10 1560000,
               11 1687826.087,
               12 1387826.087/;

 parameter R1(p)/1 750000,
                2 750000,
                3 750000/;
scalar k;
k = 733333;
$onecho > Month1.txt
dset=j   rng=a1      rdim=1
dset=d   rng=a1      cdim=1
par=c    rng=a1      rdim=1      cdim=1
$offecho

$call GDXXRW Month2.xlsx @Month1.txt
$GDXIN Month2.gdx
$load j,d
$load C
$GDXIN

variables
Z
Z1
Z2
Z3
binary variable Y(d,p,j)
binary variable X(p)
binary variable T(p)
Equations
objectiveFunction
objectiveFunction1
objectiveFunction2
objectiveFunction3
co1(p)
co2(p)
co5(p)
co6(p)
co3(d,j)
co4(d,j);

objectiveFunction1     .. Z1 =e= T("1")*sum((d,j),Y(d,"1",j)*c(j,d))-X("1")*k;
objectiveFunction2     .. Z2 =e= T("2")*sum((d,j),Y(d,"2",j)*c(j,d))-X("2")*k;
objectiveFunction3     .. Z3 =e= T("3")*sum((d,j),Y(d,"3",j)*c(j,d))-X("3")*k;
objectiveFunction      .. Z  =e= abs(Z1)+abs(Z2)+abs(Z3);
co3(d,j)               .. sum(p,y(d,p,j)) =g= 1;
co4(d,j)               .. sum(p,y(d,p,j)) =l= 1;
co1(p)                    .. X(p) =g= 1;
co2(p)                    .. X(p) =l= 1;
co5(p)                    .. T(p) =g= 1;
co6(p)                    .. T(p) =l= 1;
model supplychain1 /all/;
option optca = 0
solve supplychain1 using MINLP minimizing Z;


display r,c,k;

$ontext
791578.9474
$offtext
