year=${1}
for i in {0..79}; do for j in {-4..-4}; do echo /data/t3home000/bmaier/CASE/MC/BB_20220505/BB_batch_$i.h5 $j $year; done; done | xargs -n3 -P2 -i echo "source /home/bmaier/cms/coffea/miniconda3/etc/profile.d/conda.sh; conda activate case; python3 /home/bmaier/cms/CASE_20220505/preprocessing/cut_sb_mc.py {}; cat tmplock.txt | xargs -n1 -P1 rm -f;"

