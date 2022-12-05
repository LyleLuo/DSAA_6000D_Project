source /home/luoweile/software/spack/share/spack/setup-env.sh
spack load cuda@11.4
export CPATH=`spack location -i boost`/include:$CPATH
export PYTHONPATH=/home/luoweile/project/DSAA_6000D_Project/ads_int:$PYTHONPATH