import numpy as np
import subprocess
import yasara
from yasara import foldx_abspath
import os
if os.path.basename(os.getcwd()) != 'feature_extraction': os.chdir('./feature_extraction/')
from utils.utils import opsys, mkDir, get_mutation_list_from_inputfile, split_mutation, get_mutstr, findProcess, exit_program
if opsys == 'Windows': yasara_process_name = 'YASARA.exe'
else: yasara_process_name = 'yasara'



def yasara_mutate_residue(
        mutation,
        input_pdb_fname,
        log_fname,
        input_dir,
        output_dir,
        ff='AMBER15FB',
        move='all',
        mvdist=8,
        rep=1,
        surfout=0.65,
        surf=65,
        cntions=1,
        JToUnit=1.43932620865201e20,
):
    # format mutation / mutant
    if isinstance(mutation, list):
        mutant = '-'.join(mutation)
    else:
        mutant = mutation
        mutation = [mutation]

    # set parameters
    surfcost = (surfout) / 6.02214199e20 * JToUnit
    log_fpath = output_dir + log_fname

    # initialize lists for storing calculated values
    yasaraPDB_opt = None
    ebindDDG_opt = None

    # initialize YASARA
    print('Started YASARA')
    yasara.info.mode = 'txt'
    yasara.info.licenseshown = 0
    yasara.Console('Off')
    yasara.EnergyUnit('kcal/mol')
    yasara.Clear()

    # load input structure
    cpx = input_dir + input_pdb_fname
    yasara.LoadPDB(cpx)
    yasara.Console('Off')
    yasara.CleanAll()
    yasara.CellAuto(extension='10')
    yasara.ForceField(ff, setpar='yes')
    yasara.FixAll()
    print('Loaded PDB')

    # get mutation to WT AA and MT residue
    for mutgrp in mutation:
        WT, position, MT = split_mutation(mutgrp)
        yasara.FreeAtom(move + ' with distance< ' + str(mvdist) + ' from res ' + str(position))

    for n in range(rep):
        res = {}
        setname = '{0}_{1}_{2}_d{3}_s{4}_c{5}_r{6}'.format(mutant, ff, move, mvdist, surf, cntions, n)
        yasara.RandomSeed(1234 * n)
        for WT_or_MT in ['WT','MT']:
            # iterate through mutations
            for mutgrp in mutation:
                WT, position, MT = split_mutation(mutgrp)
                selection = str(position)
                if WT_or_MT=='WT':
                    aa_for_swapres = WT
                elif WT_or_MT=='MT':
                    aa_for_swapres = MT
                # swap residue
                yasara.SwapRes(selection, aa_for_swapres)
                # optimize residue
                if aa_for_swapres != 'Gly' and aa_for_swapres != 'Ala':
                    yasara.OptimizeRes(aa_for_swapres + ' ' + selection, method='SCWALL')
                    yasara.Boundary(Type='Periodic')
                print('Performed SwapRes ', selection, aa_for_swapres, 'for '+WT_or_MT+' structure')

            # minimize energy
            yasara.ExperimentMinimization(convergence=0.01)
            yasara.Experiment('On')
            yasara.Wait('ExpEnd')
            yasara.ChargeObj('All', 0)

            # get energies
            epotList = yasara.Energy('All') #
            res['epot'+WT_or_MT] = sum(epotList)
            res['esolcol'+WT_or_MT], res['esolvdw'+WT_or_MT] = yasara.SolvEnergy(method='BoundaryFast')
            yasara.Sim('On')
            surfaccList = yasara.Surf('Accessible')
            res['surfacc'+WT_or_MT]  = surfaccList.pop()
            # calculate total energies
            res['esol'+WT_or_MT] = res['esolcol'+WT_or_MT] + res['esolvdw'+WT_or_MT] + res['surfacc'+WT_or_MT] * surfcost
            res['ebinding'+WT_or_MT] = res['epot'+WT_or_MT] + res['esol'+WT_or_MT]

            if WT_or_MT=='MT':
                yasara.AddObj('All')
            yasara.Sim('Off')

        # calculate DDG
        res['ebindingDDG'] = res['ebindingWT'] - res['ebindingMT']

        # save mutated PDB
        PDBfile_mutated = output_dir + setname + ".pdb"
        PDBname = setname + ".pdb"
        if n == 0 or res['ebindingDDG'] < ebindDDG_opt:
            ebindDDG_opt = res['ebindingDDG']
            yasaraPDB_opt = PDBname
            yasara.SavePDB(1, PDBfile_mutated)
            print("New Lowest: " + str(ebindDDG_opt))
        else:
            print("Same Lowest: " + str(ebindDDG_opt))
        yasara.LogAs(log_fpath, 'yes')
        yasara.Print('{0} {1}'.format(setname, ebindDDG_opt))

def foldxRepair(pdbs, output_dir, input_dir, foldx_path):
    print('Started Foldx Repair PDB')
    indv_list_fname = os.path.join(output_dir, 'pdbList.txt')
    with open(indv_list_fname, 'w') as output:
        if isinstance(pdbs, list):
            output.write('{0};\n'.format(';\n'.join(pdbs)))
        else:
            output.write('{0};\n'.format(pdbs))

    cmd = [
        foldx_path,
        '--command=RepairPDB',
        '--pdb-list={0}'.format(indv_list_fname),
        '--pdb-dir={0}'.format(input_dir),
        '--out-pdb=true',
        '--output-dir={0}'.format(output_dir)]

    foldx_args = " ".join(cmd)
    print(foldx_args)
    subprocess.call(cmd)
    print('Ended Foldx Repair PDB')

def foldxBuild(mutation, pdb, output_dir, input_dir, foldx_path, receptor_molname='R', reverse_mutation=True):
    mut_to_build = []
    # convert single mutation to list of that mutation
    if not isinstance(mutation, list):
        mutation = [mutation]
    # reverse mutation
    if reverse_mutation:
        for mutgrp in mutation:
            WT, pos, MT = split_mutation(mutgrp, aa_letter_representation=True)
            revMut = MT + receptor_molname + str(pos) + WT
            mut_to_build.append(revMut)
        print('Mutations reversed:', mutation, '>>', mut_to_build)
    else:
        for mutgrp in mutation:
            WT, pos, MT = split_mutation(mutgrp, aa_letter_representation=True)
            Mut = WT + receptor_molname + str(pos) + MT
            mut_to_build.append(Mut)
        print('Mutations not reversed:', mutation, '>>', mut_to_build)
    print('Started Foldx BuildModel. Building ' + ','.join(mut_to_build))
    indv_list_fname = os.path.join(output_dir, 'individual_list.txt')
    with open(indv_list_fname, 'w') as output:
        if isinstance(mut_to_build, list):
            output.write('{0};\n'.format(','.join(mut_to_build)))
        else:
            output.write('{0};\n'.format(mut_to_build))

    cmd = [
        foldx_path,
        '--command=BuildModel',
        '--pdb={0}_Repair.pdb'.format(pdb),
        '--mutant-file={0}'.format(indv_list_fname),
        '--pdb-dir={0}'.format(input_dir),
        '--ionStrength=0.05',
        '--pH=7',
        '--vdwDesign=2',
        '--out-pdb=true',
        '--numberOfRuns=5',
        '--temperature=298',
        '--moveNeighbours=true',
        '--output-dir={0}'.format(output_dir)]

    subprocess.call(cmd)
    print('Ended Foldx BuildModel')

def run_stability_pipeline(mutation, input_pdb_fname, pdb_dir, swapRes_dir, repair_dir, build_dir, log_fname, receptor_molname='R', remove_existing_dir=True, foldx_abspath=os.path.abspath('../../../yasara/foldx_2025/foldx_20251231_linux')):

    #  make directories
    mutstr, mutation = get_mutstr(mutation)
    SwapRes_path = mkDir(mutstr, swapRes_dir, remove_existing_dir) + '/'
    Repair_path = mkDir(mutstr, repair_dir, remove_existing_dir) + '/'
    Build_path = mkDir(mutstr, build_dir, remove_existing_dir) + '/'
    print('Save folders:', SwapRes_path, Repair_path, Build_path)

    # get yasara parameters
    ff = 'AMBER15FB'
    move = 'all'
    mvdist = 8
    surf = 65
    cntions = 1
    setname = '{0}_{1}_{2}_d{3}_s{4}_c{5}_r{6}'.format('-'.join(mutation), ff, move, mvdist, surf, cntions, 0)
    swapres_output_pdb = setname + '.pdb'
    foldx_repair_inputPDB = swapres_output_pdb
    foldx_build_inputPDB = setname + '_Repair.pdb'

    ## RUN YASARA MUTAGENESIS ##
    yasara_mutate_residue(mutation, input_pdb_fname, log_fname, pdb_dir, SwapRes_path)
    print('Finished running YASARA mutagenesis on '+mutstr)

    ## RUN FOLDX REPAIR ##
    foldxRepair(foldx_repair_inputPDB, os.path.abspath(Repair_path), os.path.abspath(SwapRes_path), foldx_abspath)
    print('Finished running foldxRepair on '+mutstr)

    ## RUN FOLDX BUILD ##
    foldxBuild(mutation, foldx_build_inputPDB.replace('_Repair.pdb',''), os.path.abspath(Build_path), os.path.abspath(Repair_path), foldx_abspath, receptor_molname)
    print('Finished running foldxBuild on '+mutstr)


def get_yasara_foldx_stability_features():
    # define directories & filenames
    input_dir = '../data/feature_extraction/Input/'
    pdb_dir = input_dir
    output_dir = '../data/feature_extraction/'
    input_fname = 'GOh1052_mutPos_DomainIII.txt'
    input_pdb_fname = 'YASARA_2EIE_GOh1052.pdb'
    output_fname = 'DDGstability_GOh1052.csv'
    log_fname = input_pdb_fname.strip('.pdb')
    swapRes_dir = output_dir + 'SwapRes/'
    repair_dir = output_dir + 'Repair/'
    build_dir = output_dir + 'Build/'
    remove_existing_dir = False
    receptor_molname = 'A'

    # get mutations
    mutations, res_mut_dict = get_mutation_list_from_inputfile(input_fname, input_dir)
    # mutations = []
    print('# of positions to mutate:', len(res_mut_dict), list(res_mut_dict.keys()))
    print('# of mutations:', len(mutations), mutations)


    ##############################
    ## RUN STABILITY PROCESSING ##
    ##############################
    for i, mutation in enumerate(mutations):
        run_stability_pipeline(mutation, input_pdb_fname, pdb_dir, swapRes_dir, repair_dir, build_dir, log_fname, receptor_molname,
                               remove_existing_dir=remove_existing_dir, foldx_abspath=foldx_abspath)
    # exit yasara after processing all mutations
    yasara_pid = findProcess(yasara_process_name)
    if len(yasara_pid)>0:
        exit_program(yasara_pid[0])


    # combine files spawned
    print('Combining stability calculation results for all mutants...')
    stability_cols = ['total energy', 'Backbone Hbond', 'Sidechain Hbond', 'Van der Waals', 'Electrostatics',
                      'Solvation Polar', 'Solvation Hydrophobic', 'Van der Waals clashes', 'entropy sidechain',
                      'entropy mainchain', 'sloop_entropy', 'mloop_entropy', 'cis_bond', 'torsional clash',
                      'backbone clash', 'helix dipole', 'water bridge', 'disulfide', 'electrostatic kon',
                      'partial covalent bonds', 'energy Ionisation', 'Entropy Complex']
    cols = ['Mutation', 'DDG', 'DDG_foldx'] + [col+'_WT' for col in stability_cols] + [col+'_MT' for col in stability_cols]

    mutations_list = []
    DDG_foldx_list = []
    DDG_list = []
    wt_features_list = []
    mt_features_list = []

    for root, dirs, files in os.walk(build_dir):
        for name in files:
            if name.startswith("Raw"):
                mut = name.split("_")[1]
                mutations_list.append(mut)
                with open(os.path.join(root, name)) as f:
                    # parse raw data
                    lines = f.read().splitlines()
                    header = lines[8].split('\t')
                    data = [l.split('\t') for l in lines[9:]]
                    # get mutant features
                    data_mt = np.array([row[1:] for row in data[:5]]).astype(float)
                    data_mt_avg = np.mean(data_mt,axis=0)
                    mt_features_list.append(list(data_mt_avg))
                    # get wildtype features
                    data_wt = np.array([row[1:] for row in data[5:]]).astype(float)
                    data_wt_avg = np.mean(data_wt, axis=0)
                    wt_features_list.append(list(data_wt_avg))
                    # get DDG
                    DDG_foldx = data_mt_avg[0] - data_wt_avg[0]
                    DDG_foldx_list.append(DDG_foldx)
                    DDG_list.append(-DDG_foldx)
                    print(name, 'DDG:', DDG_list[-1])

    txt_all_list = [','.join(cols)]
    for mut, DDG, DDG_foldx, wt_features, mt_features in zip(mutations_list, DDG_list, DDG_foldx_list, wt_features_list, mt_features_list):
        txt_all_list.append(','.join(list(map(str, [mut, DDG, DDG_foldx]+wt_features+mt_features))))

    if os.path.exists(output_dir + output_fname + '.csv'):
        write_mode = 'a'
        txt_all_list = txt_all_list[1:]
    else:
        write_mode = 'w'

    # get text string to write
    txt_all = '\n'.join(txt_all_list)
    txt_all = txt_all.replace('\n\n', '\n').replace(',\n', '\n')

    # update or save file
    with open(output_dir + output_fname, write_mode) as f:
        f.write(txt_all)

if __name__ == "__main__":  # confirms that the code is under main function
    get_yasara_foldx_stability_features()
