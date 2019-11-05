"""
Created at 20/11/18 16:18
@author: gregor 
"""
import os
import numpy as np
import pandas as pd


class CombinedPrinter(object):
    """combined print function.
    prints to logger and/or file if given, to normal print if non given.

    """
    def __init__(self, logger=None, file=None):

        if logger is None and file is None:
            self.out = [print]
        elif logger is None:
            self.out = [print, file.write]
        elif file is None:
            self.out = [print, logger.info]
        else:
            self.out = [print, logger.info, file.write]

    def __call__(self, string):
        for fct in self.out:
            fct(string)

def spec_to_id(spec):
    """Get subject id from string"""
    return int(spec[-5:])


def pat_roi_GS_histo_check(root_dir):
    """ Check, in histo files, whether patient-wide Gleason Score equals maximum GS found in single lesions of patient.
    """

    histo_les_path = os.path.join(root_dir, "MasterHistoAll.csv")
    histo_pat_path = os.path.join(root_dir, "MasterPatientbasedAll_clean.csv")

    with open(histo_les_path,mode="r") as les_file:
        les_df = pd.read_csv(les_file, delimiter=",")
    with open(histo_pat_path, mode="r") as pat_file:
        pat_df = pd.read_csv(pat_file, delimiter=",")

    merged_df = les_df.groupby('Master_ID').agg({'Gleason': 'max', 'segmentationsNameADC': 'last'})

    for pid in merged_df.index:
        merged_df.set_value(pid, "GSBx", pat_df[pat_df.Master_ID_Short==pid].GSBx.unique().astype('uint32'))

    #print(merged_df)
    print("All patient-wise GS are maximum of lesion-wise GS?", np.all(merged_df.Gleason == merged_df.GSBx), end="\n\n")
    assert np.all(merged_df.Gleason == merged_df.GSBx)


def lesion_redone_check(root_dir, out_path=None):
    """check how many les annotations without post_fix _Re exist and if exists what their GS is
    """

    histo_les_path = os.path.join(root_dir, "Dokumente/MasterHistoAll.csv")
    with open(histo_les_path,mode="r") as les_file:
        les_df = pd.read_csv(les_file, delimiter=",")
    if out_path is not None:
        out_file = open(out_path, "w")
    else:
        out_file = None
    print_f = CombinedPrinter(file=out_file)

    data_dir = os.path.join(root_dir, "Daten")

    matches = {}
    for patient in [dir for dir in os.listdir(data_dir) if dir.startswith("Master_") \
                    and os.path.isdir(os.path.join(data_dir, dir))]:
        matches[patient] = {}
        pat_dir = os.path.join(data_dir,patient)
        lesions = [os.path.splitext(file)[0] for file in os.listdir(pat_dir) if os.path.isfile(os.path.join(pat_dir,file)) and file.startswith("seg") and "LES" in file]
        lesions_wo = [os.path.splitext(file)[0] for file in lesions if not "_Re" in file]
        lesions_with = [file for file in lesions if "_Re" in file and not "registered" in file]

        matches[patient] = {les_wo : [] for les_wo in lesions_wo}

        for les_wo in matches[patient].keys():
            matches[patient][les_wo] += [les_with for les_with in lesions_with if les_with.startswith(les_wo)]

    missing_les_count = 0
    for patient, lesions in sorted(list(matches.items())):
        pat_df = les_df[les_df.Master_ID==spec_to_id(patient)]
        for les, les_matches in sorted(list(lesions.items())):
            if len(les_matches)==0:
                if "t2" in les.lower():
                    les_GS = pat_df[pat_df.segmentationsNameT2==les]["Gleason"]
                elif "adc" in les.lower():
                    les_GS = pat_df[pat_df.segmentationsNameADC==les]["Gleason"]
                if len(les_GS)==0:
                    les_GS = r"[no histo finding!]"
                print_f("Patient {}, lesion {} with GS {} has no matches!\n".format(patient, les, les_GS))
                missing_les_count +=1
            else:
                del matches[patient][les]
            #elif len(les_matches) > 1:
            #    print("Patient {}, Lesion {} has {} matches: {}".format(patient, les, len(les_matches), les_matches))
        if len(matches[patient])==0:
            del matches[patient]

    print_f("Total missing lesion matches: {} within {} patients".format(missing_les_count, len(matches)))

    out_file.close()


if __name__=="__main__":

    #root_dir = "/mnt/HDD2TB/Documents/data/prostate/data_di_ana_081118_ps384_gs71/histos/"
    root_dir = "/mnt/E132-Projekte/Move_to_E132-Rohdaten/Prisma_Master/Dokumente"
    pat_roi_GS_histo_check(root_dir)

    root_dir = "/mnt/E132-Projekte/Move_to_E132-Rohdaten/Prisma_Master"
    out_path = os.path.join(root_dir,"lesion_redone_check.txt")
    lesion_redone_check(root_dir, out_path=out_path)

