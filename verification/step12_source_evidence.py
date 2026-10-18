"""공식 설명·공개 소스·로컬 참고문헌의 출처 및 해시를 별도로 기록한다.

OpenSees 핵심 3.5.1의 배포 바이너리와 v3.5.0 공개 소스의 동일 커밋을 보증하지 않는다.
실제 설치 바이너리의 동작은 step12_shell_audit의 실행시험으로 검증한다.
"""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from verification.step12_shell_audit import ROOT, OUT, save_json


def run():
    directory=OUT/"sources"
    directory.mkdir(parents=True,exist_ok=True)
    urls={
        "ElasticMembranePlateSection_v3.5.0.cpp":"https://raw.githubusercontent.com/OpenSees/OpenSees/v3.5.0/SRC/material/section/ElasticMembranePlateSection.cpp",
        "ShellMITC4_v3.5.0.cpp":"https://raw.githubusercontent.com/OpenSees/OpenSees/v3.5.0/SRC/element/shell/ShellMITC4.cpp",
        "section_documentation.html":"https://openseespydoc.readthedocs.io/en/latest/src/elasticMembranePlateSection.html",
        "shell_documentation.html":"https://openseespydoc.readthedocs.io/en/latest/src/ShellMITC4.html",
    }
    records=[]
    for filename,url in urls.items():
        try:
            content=urlopen(Request(url,headers={"User-Agent":"Mozilla/5.0"}),timeout=30).read()
            (directory/filename).write_bytes(content)
            records.append({"url":url,"file":filename,"sha256":hashlib.sha256(content).hexdigest()})
        except (HTTPError,URLError) as error:
            # 직접 내려받기가 제한되어도 웹 도구로 열람한 공식 URL은 출처로 보존한다.
            records.append({"url":url,"file":None,"download_error":str(error),
                            "inspection":"이번 검토에서 웹 도구로 본문 확인; 로컬 원문 사본은 없음"})
    paper=next((ROOT/"reference").glob("3_*.pdf"))
    save_json(directory/"provenance.json",{
        "retrieved_utc":datetime.now(timezone.utc).isoformat(),"sources":records,
        "version_limit":"공개 v3.5.0 소스는 구현 원리의 보조 근거. 설치 3.5.1 바이너리의 정확한 빌드 커밋은 확인되지 않음.",
        "paper":{"path":paper.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(paper.read_bytes()).hexdigest(),
                 "doi":"10.1016/j.engstruct.2015.03.025","read_pages":"Appendix A: PDF 13~14쪽, 인쇄 82~83쪽"}})


if __name__=="__main__":
    run()
