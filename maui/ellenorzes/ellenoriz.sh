#!/bin/bash
# A MAUI kliens fordítás-ellenőrzése MAUI workload nélkül (lásd ellenorzes.csproj).
set -e
cd "$(dirname "$0")"
rm -rf src && mkdir src
(cd ../CLOUDtacticAnalyzer && tar cf - --exclude=Platforms --exclude=bin --exclude=obj --exclude='*.csproj' .) | tar xf - -C src
dotnet build ellenorzes.csproj -nologo -v q "$@"
