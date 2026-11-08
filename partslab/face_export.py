"""Deterministic source-bound face responses, never retunes the frozen bases."""
from .io import ROOT, read, digest
from .data import load
from .face import response, verify


def export():
    models = read(ROOT/'artifacts/models.json')
    nominated = read(ROOT/'artifacts/certificate-candidates.json')['cases']
    pixels = load('test')
    cases = []
    for rank in [8,16,32]:
        for run in [0,1]:
            for row in [0,1,2]:
                candidate = next(c for c in nominated if (c['rank'],c['run'],c['row'])==(rank,run,row))
                support = [k for k,v in enumerate(candidate['coefficients']) if v>0]
                for pixel in [18,27,36]:
                    witness = response(pixels[row].tolist(),models['arms'][str(rank)]['runs'][run]['basis'],support,pixel)
                    if not verify(witness):
                        raise ArithmeticError('invalid exported response')
                    cases.append({'rank':rank,'run':run,'row':row,'pixel':pixel,'witness':witness})
    controls = [response([.5,.5],[[1,1],[1,0]],[0],0),
                response([0,0],[[1,0]],[],0),response([.5],[[1],[1]],[0],0),
                response([.5,.5],[[1,0],[0,1]],[0,1],0)]
    return {'schema':1,'models_sha256':digest(models),
            'candidate_sha256':digest(read(ROOT/'artifacts/certificate-candidates.json')),
            'cases':cases,'controls':controls}
