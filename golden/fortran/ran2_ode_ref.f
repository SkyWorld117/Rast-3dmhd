C**********************************************************************
C  Reference values for RAN2 and the static stratification ODE, used to
C  validate the C++ ports of ran2.cpp / ode.cpp.
C
C  RAN2: seed IDUM=-062659, print draws at fixed stride.
C  Static profile: same integration STATIC performs for the default
C  (non-LREM) case, for NPZ=32 (the golden geometry).
C**********************************************************************
        PROGRAM RAN2ODEREF
        INCLUDE '3dmhdparam.f'
        DIMENSION IIR(97)
        DIMENSION Y(2),DYDX(2)
        DIMENSION T(NPZ),R(NPZ)
        COMMON/CPAR/CV,OCV,ORE,RE,REPR,THETA,GRAV,AMPT,SF,GAMMA
        COMMON/CPEN/PZP,SIGMA,RKAPST,TB,RKAPA,DKAPA,RKAPM
        COMMON/BOUNDS/XMAX,YMAX,ZMAX
        DIMENSION RKAPA(NZ),DKAPA(NZ)
C
        THETA=0.25D0
        GRAV=0.625D0
        PZP=0.0D0
        SIGMA=0.0D0
        RKAPST=(0.0+1.0)*THETA/GRAV
        XMAX=2.0D1
        YMAX=2.0D1
        ZMAX=4.0D1
C----------------------------------------------------------------------
C  RAN2 (same call shape as STATIC: 1.0+AMPT*(RAN2-0.5) applied per cell).
C----------------------------------------------------------------------
        IDUM=-062659
        IIY=0
        K=0
        VAL=0.0D0
        DO I=1,30000
          R2=RAN2(IDUM,IIY,IIR)
          IF (MOD(I,1000).EQ.0) THEN
            K=K+1
            WRITE(*,'(A5,I5,A1,E27.18)')'RAN2 ',I,' ',R2
          ENDIF
        END DO
C----------------------------------------------------------------------
C  Static T/R profile (exactly the STATIC ODE loop).
C----------------------------------------------------------------------
        DZZ=1.0E00/FLOAT(NPZ-1)
        EPS=1.0E-12
        T(1)=1.0E00
        R(1)=1.0E00
        ZZ=0.0E00
        FFZ=0.0E00
        PLN=0.0E00
        Y(1)=FFZ
        Y(2)=PLN
        DO 10 K=2,NPZ
          HTRY=DZZ
          CALL DERIVS(ZZ,Y,2,DYDX)
          CALL BSSTEP(Y,DYDX,2,ZZ,HTRY,EPS,HDID,HNEXT)
          IF (HDID.NE.HTRY) THEN
            WRITE(*,*)'STATIC: Static structure error',HDID,HTRY
            CALL MPI_FINALIZE(IERR)
            STOP
          ENDIF
          T(K)=1.0E00+Y(1)
          R(K)=EXP(Y(2))/(1.0E00+Y(1))
10      CONTINUE
        WRITE(*,'(A5,E27.18)')'T(1) ',T(1)
        WRITE(*,'(A5,E27.18)')'T(4) ',T(4)
        WRITE(*,'(A5,E27.18)')'T(10)',T(10)
        WRITE(*,'(A5,E27.18)')'T(20)',T(20)
        WRITE(*,'(A5,E27.18)')'T(32)',T(32)
        WRITE(*,'(A5,E27.18)')'R(32)',R(32)
        WRITE(*,'(A5,E27.18)')'RB   ',R(NPZ)
        END
