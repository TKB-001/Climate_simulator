from pyscf import gto, scf
from pyscf.prop import polarizability
from pyscf import tdscf
import hapi as h
import logging as l
from config import GAS_CONFIG, os
import numpy as np
from pyscf.scf import addons 
from chemicals import permittivity  
import shutil


'''PySCF: Sun, Q., et al. (2018). PySCF: the Python-based simulations of chemistry framework. WIREs Computational Molecular Science.
 R.V. Kochanov, I.E. Gordon, L.S. Rothman, P. Wcislo, C. Hill, J.S. Wilzewski,
           HITRAN Application Programming Interface (HAPI): A comprehensive approach
           to working with spectroscopic data, J. Quant. Spectrosc. Radiat. Transfer 177, 15-30 (2016)
           DOI: 10.1016/j.jqsrt.2016.03.005'''


l.basicConfig(

    filename='chemistry.log', 
    level=l.INFO,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s' 
)

l.info('check')


def RHF_obj(mol):
    mf = scf.RHF(mol)
    mf.kernel()   
    return mf      



def run_tddft(mf,
              nstates=10,
              ld_threshold=1e-6,
              min_eig_cutoff=1e-8):

    mf = addons.remove_linear_dep_(mf, threshold=ld_threshold)
    mf.conv_tol = 1e-9
    mf.diis_space = 12
    mf.kernel()

    stab, info = mf.stability()
    if isinstance(stab, bool):
        unstable = not stab
    else:
        unstable = stab.min() < -1e-8
    if unstable:
        l.info("SCF unstable:re-optimizing with Newton")
        mf = scf.newton(mf).run()

    td = tdscf.TDHF(mf)  
    td.nstates = nstates

    def peek_eigs(A, B=None):

        if A.ndim == 4:
            nocc, nvir = A.shape[0], A.shape[1]
            A2 = A.reshape(nocc*nvir, nocc*nvir)
            B2 = B.reshape(nocc*nvir, nocc*nvir) if B is not None else 0
            M = A2 + B2
        else:
        
            M = A + (B if B is not None else 0)

        w = np.linalg.eigvalsh(M)
        l.debug("  Casida eigs (lowest five): %s", w[:5])
        return w

    A, B = td.get_ab()
    w = peek_eigs(A, B)

    if w[2] < min_eig_cutoff: #O2 can have misnicule or negative eigenvalues, and I'm fairly certian that other gases may have the same problem
        l.warning(
            "3rd Casida eig = %.3e < %.1e; skipping full TDDFT in favor of TDA",
            w[2], min_eig_cutoff
        )
        td.tamm_dancoff = True
        peek_eigs(A, None)
        td.kernel()
        l.info("TDA succeeded")
        return td

    try:
        td.kernel()
        l.info("Full TDDFT succeeded")
    except np.linalg.LinAlgError as e:
        l.warning("Full TDDFT failed (%s); falling back to TDA", e)
        tda = tdscf.TDA(mf)
        tda.nstates = nstates
        peek_eigs(A, None)
        tda.kernel()
        l.info("TDA succeeded")
        return tda
    except Exception as fallback_error:
        l.error("Both TDDFT and TDA failed (skipping molecule): %s", fallback_error)
        return None


    return td



def begin():
    global isoto_numbers, cas_numbers, permittivity_data, depolarization
    mols = {
    name: gto.M(atom=info['atom'],
                basis=info['basis'],
                unit=info['unit'])
    for name, info in GAS_CONFIG.items()
    }
    rhf_results = {
    name: RHF_obj(mol)
    for name, mol in mols.items()
    }      
    tdhf_results = {
    name: run_tddft(mf, nstates=10)
    for name, mf in rhf_results.items()
    }
    pol_tensors = {}
    for name, mf in rhf_results.items():
        p = polarizability.rks.Polarizability(mf)
        p.kernel()
        pol_tensors[name] = p.polarizability() * 1.482e-31



    pol_calc_av = sum(pol_tensors.values()) / len(pol_tensors)

    permittivity_data = permittivity.permittivity_data_CRC

    cas_numbers = {gas: data['cas_number'] for gas, data in GAS_CONFIG.items()}

    for compound, cas in cas_numbers.items():
        try:
            compound_data = permittivity_data.loc[cas]
            l.info(f"\n{compound} (CAS {cas}):")
            l.info(compound_data)
        except KeyError:
            l.warning(f"\nNo permittivity data found for {compound} (CAS {cas}).")

    if os.path.isdir('data'):
        shutil.rmtree('data')
    h.db_begin('data')
    evals, evecs = np.linalg.eig(pol_calc_av)
    alpha_parallel = np.max(evals)
    alpha_perp = np.mean(evals[evals != np.max(evals)])
    polarizability_mol = (alpha_parallel+2*alpha_perp)/3
    gamma = alpha_parallel-alpha_perp
    depolarization = (6*gamma**2)/((45*polarizability_mol**2)+(7*gamma **2))

    isoto_numbers = {gas: data['I'] for gas, data in GAS_CONFIG.items()}

    for compound, isoto in isoto_numbers.items():
        try:
            h.fetch(compound,isoto,1,1,3500, Parameters=['nu', 'Sw']) #figure this out
        except Exception as e:
            l.exception(f"Failed to fetch data for {compound} (isotope {isoto}): {e}")
    return isoto_numbers,  cas_numbers, permittivity_data, depolarization, polarizability_mol

