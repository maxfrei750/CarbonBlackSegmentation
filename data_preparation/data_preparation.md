1. Aggregated all ground truth files in one folder and all images in another one: 
    `mv */* .`
2. Converted from tif to png:
    `mogrify -format png *.*`
3. Removed outliers:
    * Case2_E9_032
    * Case2_E9_033
    * Case4_D8_001
4. Removed images with incomplete ground truths (probably operator fatigue):
    * Case2_E9_019
    * Case2_E9_034
    * Case2_E9_035
    * Case4_D8_034
    * Case5_D10_023
5. Removed images with incomplete ground truths, caused by agglomerates touching the image border:
    * Case1_E7_002
    * Case1_E7_031
    * Case3_D6_007
    * Case3_D6_025
    * Case3_D6_041
    * Case4_D8_011
    * Case4_D8_033
    * Case5_D10_005
    * Case5_D10_024
    * Case6_C7_003
    * Case6_C7_010
    * Case6_C7_024
    * Case6_C7_039
    * Case1_E7_003
    * Case1_E7_036
    * Case2_E9_042
    * Case3_D6_008
    * Case3_D6_026
    * Case3_D6_042
    * Case4_D8_012
    * Case5_D10_006
    * Case5_D10_027
    * Case6_C7_004
    * Case6_C7_013
    * Case6_C7_025
    * Case6_C7_041
    * Case1_E7_004
    * Case2_E9_005
    * Case3_D6_002
    * Case3_D6_010
    * Case3_D6_027
    * Case4_D8_002
    * Case4_D8_014
    * Case4_D8_037
    * Case5_D10_007
    * Case5_D10_036
    * Case6_C7_006
    * Case6_C7_014
    * Case6_C7_026
    * Case1_E7_007
    * Case2_E9_007
    * Case3_D6_003
    * Case3_D6_013
    * Case3_D6_028
    * Case4_D8_005
    * Case4_D8_024
    * Case4_D8_043
    * Case5_D10_009
    * Case5_D10_045
    * Case6_C7_007
    * Case6_C7_019
    * Case6_C7_030
    * Case1_E7_025
    * Case2_E9_008
    * Case3_D6_004
    * Case3_D6_024
    * Case3_D6_037
    * Case4_D8_006
    * Case4_D8_027
    * Case5_D10_002
    * Case5_D10_013
    * Case6_C7_001
    * Case6_C7_009
    * Case6_C7_023
    * Case6_C7_033
6. Split data into training and test sets (85%/15%): 

    `python split_data.py`
