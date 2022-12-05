year=${1}
for i in /data/t3home000/bmaier/CASE/NEWMC/signals/2018/*h5; do for j in {5..5}; do echo $i $j $year; done; done | xargs -n3 -P2 -i echo "source /home/bmaier/cms/coffea/miniconda3/etc/profile.d/conda.sh; conda activate case; python3 /home/bmaier/cms/CASE_20220505/preprocessing/cut_srsb.py {}; cat tmplock.txt | xargs -n1 -P1 rm -f;"

