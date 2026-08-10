C**********************************************************************
C  GSTATE - golden state dump for validating the C++ rewrite.
C  Self-describing binary format (little-endian, STREAM access, no
C  record markers), so the C++ reader can parse it exactly.
C
C  Layout:
C    int32 magic   = 1129928788 ('GDST')
C    int32 nfields = 27, int32 n1d = 14
C    int32[13]     header ints : nx ny nz mype mypey mypez nit lmag
C                                ixcon iycon izcon itcon ibcon
C    f64[26]       header reals: dt timc timt timi umach re repr cv ocv
C                                ore theta grav rkapst orm rm obeta omx
C                                omz sf ampt dd hx h2x hy h2y hz h2z
C    then nfields blocks: int64 count, f64[count]  (count = nx*ny*nz)
C    then n1d blocks:     int64 count, f64[count]
C    int64 endmark = -1
C  Within a block elements are written in Fortran storage order, which is
C  x-fastest (matches C++ flat index i + nx*(j + ny*k)).
C**********************************************************************
        SUBROUTINE GSTATE(TAG,ITER)
C
        INCLUDE '3dmhdparam.f'
C
        CHARACTER*(*) TAG
        INTEGER ITER
C
        DIMENSION RU(NX,NY,NZ),RV(NX,NY,NZ),RW(NX,NY,NZ),RO(NX,NY,NZ)
     2           ,TT(NX,NY,NZ)
        DIMENSION UU(NX,NY,NZ),VV(NX,NY,NZ),WW(NX,NY,NZ)
        DIMENSION FU(NX,NY,NZ),FV(NX,NY,NZ),FW(NX,NY,NZ),FR(NX,NY,NZ)
     2           ,FT(NX,NY,NZ)
        DIMENSION ZRU(NX,NY,NZ),ZRV(NX,NY,NZ),ZRW(NX,NY,NZ)
     2           ,ZRO(NX,NY,NZ),ZTT(NX,NY,NZ)
        DIMENSION WW1(NX,NY,NZ),WW2(NX,NY,NZ),WW3(NX,NY,NZ)
        DIMENSION BX(NX,NY,NZ),BY(NX,NY,NZ),BZ(NX,NY,NZ)
        DIMENSION ZBX(NX,NY,NZ),ZBY(NX,NY,NZ),ZBZ(NX,NY,NZ)
        DIMENSION SP1(0),SP2(0),SP3(0),SP4(0),SP5(0),SP6(0),SP7(0),
     1            SP8(0),SP9(0),SP10(0),SP11(0),SP12(0),SP13(0),
     2            SP14(0),SP15(0),SP16(0),SP17(0),SP18(0),SP19(0),
     3            SP20(0),SP21(0),SP22(0),SP23(0),SP24(0),SP25(0),
     4            SP26(0)
C
        DIMENSION EXX(NX),DXXDX(NX),D2XXDX2(NX),DDX(NX)
        DIMENSION WYY(NY),DYYDY(NY),D2YYDY2(NY),DDY(NY)
        DIMENSION ZEE(NZ),DZZDZ(NZ),D2ZZDZ2(NZ),DDZ(NZ)
        DIMENSION RKAPA(NZ),DKAPA(NZ)
C
        COMMON/BIG/RU,SP1,RV,SP2,RW,SP3,RO,SP4,TT,SP5,UU,SP6,VV,SP7,WW
     2            ,SP8,FU,SP9,FV,SP10,FW,SP11,FR,SP12,FT,SP13
     3            ,ZRU,SP14,ZRV,SP15,ZRW,SP16,ZRO,SP17,ZTT
     4            ,SP18,WW1,SP19,WW2,SP20,WW3
     5            ,SP21,BX,SP22,BY,SP23,BZ,SP24,ZBX,SP25,ZBY,SP26,ZBZ
C
        COMMON/AJACOBI/EXX,DXXDX,D2XXDX2,DDX,WYY,DYYDY,D2YYDY2,DDY
     2                ,ZEE,DZZDZ,D2ZZDZ2,DDZ
        COMMON/CPAR/CV,OCV,ORE,RE,REPR,THETA,GRAV,AMPT,SF,GAMMA
        COMMON/CMAG/ORM,RM,OBETA,AMPB,BFH,BZP
        COMMON/CPEN/PZP,SIGMA,RKAPST,TB,RKAPA,DKAPA,RKAPM
        COMMON/CROT/OMX,OMZ
        COMMON/CTIM/DT,TIMT,TIMC,TIMI
        COMMON/TRACE/UMACH
        COMMON/ITER/NTOTAL,NSTEP0,NIT
        COMMON/COMMUN/MYPE,MYPEY,MYPEZ,MPISIZE
        COMMON/GRID/DD,HX,H2X,HY,H2Y,HZ,H2Z,C13,C23,C43
        COMMON/BCT/IXC,IYC,IZC,ITC,IBC
C
        CHARACTER*40 FNAME
        INTEGER*8 C8, END8
        INTEGER LMAGI
C
        WRITE(FNAME,'(A,A,A,I3.3,A,I4.4)')'gstate.',TRIM(TAG),'.',
     1                                     MYPE,'.',ITER
C
        LMAGI=0
        IF (LMAG) LMAGI=1
C
        OPEN(UNIT=77,FILE=FNAME,STATUS='UNKNOWN',ACCESS='STREAM',
     1       FORM='UNFORMATTED')
        WRITE(77) 1129928788,27,14
C  Header: integer scalars.
        WRITE(77) NX,NY,NZ,MYPE,MYPEY,MYPEZ,NIT,LMAGI,
     1            IXC,IYC,IZC,ITC,IBC
C  Header: real scalars.
        WRITE(77) DT,TIMC,TIMT,TIMI,UMACH,RE,REPR,CV,OCV,ORE,THETA,
     1            GRAV,RKAPST,ORM,RM,OBETA,OMX,OMZ,SF,AMPT,
     2            DD,HX,H2X,HY,H2Y,HZ,H2Z
C  Physics arrays (3D), count-prefixed blocks, x-fastest order.
        C8=NX*NY*NZ
        CALL GBPUT(77,RU,C8)
        CALL GBPUT(77,RV,C8)
        CALL GBPUT(77,RW,C8)
        CALL GBPUT(77,RO,C8)
        CALL GBPUT(77,TT,C8)
        CALL GBPUT(77,UU,C8)
        CALL GBPUT(77,VV,C8)
        CALL GBPUT(77,WW,C8)
        CALL GBPUT(77,FU,C8)
        CALL GBPUT(77,FV,C8)
        CALL GBPUT(77,FW,C8)
        CALL GBPUT(77,FR,C8)
        CALL GBPUT(77,FT,C8)
        CALL GBPUT(77,ZRU,C8)
        CALL GBPUT(77,ZRV,C8)
        CALL GBPUT(77,ZRW,C8)
        CALL GBPUT(77,ZRO,C8)
        CALL GBPUT(77,ZTT,C8)
        CALL GBPUT(77,WW1,C8)
        CALL GBPUT(77,WW2,C8)
        CALL GBPUT(77,WW3,C8)
        CALL GBPUT(77,BX,C8)
        CALL GBPUT(77,BY,C8)
        CALL GBPUT(77,BZ,C8)
        CALL GBPUT(77,ZBX,C8)
        CALL GBPUT(77,ZBY,C8)
        CALL GBPUT(77,ZBZ,C8)
C  1D arrays.
        CALL GBPUT1(77,EXX,NX)
        CALL GBPUT1(77,DXXDX,NX)
        CALL GBPUT1(77,D2XXDX2,NX)
        CALL GBPUT1(77,DDX,NX)
        CALL GBPUT1(77,WYY,NY)
        CALL GBPUT1(77,DYYDY,NY)
        CALL GBPUT1(77,D2YYDY2,NY)
        CALL GBPUT1(77,DDY,NY)
        CALL GBPUT1(77,ZEE,NZ)
        CALL GBPUT1(77,DZZDZ,NZ)
        CALL GBPUT1(77,D2ZZDZ2,NZ)
        CALL GBPUT1(77,DDZ,NZ)
        CALL GBPUT1(77,RKAPA,NZ)
        CALL GBPUT1(77,DKAPA,NZ)
        END8=-1
        WRITE(77) END8
        CLOSE(77)
C
        RETURN
        END
C**********************************************************************
C  Write one count-prefixed block: int64 N then the N doubles in
C  x-fastest (Fortran storage) order.
C**********************************************************************
        SUBROUTINE GBPUT(LU,U,N)
        INTEGER LU
        INTEGER*8 N
        REAL*8 U(*)
        INTEGER*8 I
        WRITE(LU) N
        DO I=1,N
          WRITE(LU) U(I)
        END DO
        RETURN
        END
C**********************************************************************
        SUBROUTINE GBPUT1(LU,U,N)
        INTEGER LU,N
        REAL*8 U(*)
        INTEGER*8 N8
        INTEGER I
        N8=N
        WRITE(LU) N8
        DO I=1,N
          WRITE(LU) U(I)
        END DO
        RETURN
        END
C**********************************************************************
