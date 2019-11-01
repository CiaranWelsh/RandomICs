model_string = f"""
    function MM(km, Vmax, S)
        Vmax * S / (km + S)
    end

    function MMWithKcat(km, kcat, S, E)
        kcat * E * S / (km + S)
    end

    function NonCompetitiveInhibition(km, ki, Vmax, n, I, S)
        Vmax * S / ( (km + S) * (1 + (I / ki)^n ) )
    end

    function NonCompetitiveInhibitionWithKcat(km, ki, kcat, E, n, I, S)
        kcat * E * S / ( (km + S) * (1 + (I / ki)^n ) )
    end

    function NonCompetitiveInhibitionWithKcatAndExtraActivator(km, ki, kcat, E1, E2, n, I, S)
        kcat * E1 * E2 * S / ( (km + S) * (1 + (I / ki)^n ) )
    end


    function MA1(k, S)
        k * S
    end

    function MA2(k, S1, S2)
        k * S1 * S2
    end

    function MA1Mod(k, S, M)
        k * S * M
    end

    function MA2Mod(k, S1, S2, M)
        k * S1 * S2 * M
    end

    function CompetitiveInhibitionWithKcat(km, ki, kcat, E, I, S)
        kcat * E * S / (km + S + ((km * I )/ ki)  )
    end

    function CompetitiveInhibition(Vmax, km, ki, I, S)
        Vmax * S / (km + S + ((km * I )/ ki)  )
    end

    function Hill(km, kcat, L, S, h)
        kcat * L * (S / km)^h  /   1 + (S / km)^h
    end

    model AktModelWithMMRateLaws()
        compartment             Cell = 1;
        var IRS1                in Cell;
        var IRS1a               in Cell;
        var IRS1pS636_639       in Cell;
        var Akt                 in Cell;
        var AktpT308            in Cell;
        var TSC2                in Cell;
        var TSC2pT1462          in Cell;
        var PRAS40              in Cell;
        var PRAS40pT246         in Cell;
        var S6K                 in Cell;
        var S6KpT389            in Cell;
        var FourEBP1            in Cell;
        var FourEBP1pT37_46     in Cell;
        var PI3K                in Cell;
        var pPI3K               in Cell;
        var pmTORC1            in Cell;
        var mTORC1cyt          in Cell;
        var mTORC1lys          in Cell;
        const Insulin          in Cell;
        const AA               in Cell;

        // global variables
        Insulin = 1;
        AA = 1

        IRS1            = 10;
        IRS1a           = 0;
        IRS1pS636_639   = 0;
        PI3K            = 10;
        pPI3K           = 0;
        AktpT308        = 0;
        Akt             = 10;
        TSC2pT1462      = 10;
        TSC2            = 0;
        pmTORC1         = 0;
        mTORC1cyt       = 10;
        mTORC1lys       = 0;
        PRAS40          = 10;
        PRAS40pT246     = 0;
        FourEBP1pT37_46 = 0;
        FourEBP1        = 10;
        S6KpT389        = 0;
        S6K             = 10;

        // kinetic parameters
        _kIRS1Act                = 0.1;
        _kIRS1Inact              = 0.1;
        _kIRS1Phos               = 0.1;
        _kPI3KPhos               = 0.1;
        _kPI3KDephos             = 0.1;
        _kAktPhos_km            = 0.1;
        _kAktPhos_kcat          = 0.1;
        _kAktDephos              = 0.1;
        _kTSC2Phos               = 0.1;
        _kTSC2Dephos             = 0.1;
        _kmTORC1cytToLys         = 0.1;
        _kmTORC1LysToCyt         = 0.1;
        _kmTORC1Phos             = 0.1;
        _kmTORC1Dephos           = 0.1;
        _kPras40Phos             = 0.1;
        _kPras40Dephos           = 0.1;
        _kFourEBP1Phos           = 0.1;
        _kFourEBP1Dephos         = 0.1;
        _kS6KPhos                = 0.1;
        _kS6KDephos              = 0.1;



        // reactions // MMWithKcat(km, kcat, S, E)
        // function CompetitiveInhibitionWithKcat(km, ki, kcat, E, I, S)
        // function MM(km, Vmax, S)
        // function NonCompetitiveInhibitionWithKcat(km, ki, kcat, E, n, I, S)
        R1f     : IRS1 => IRS1a                         ; Cell * _kIRS1Act*IRS1*Insulin;
        R2f     : IRS1a => IRS1pS636_639                ; Cell * _kIRS1Phos*IRS1a*S6KpT389
        R2b     : IRS1pS636_639 => IRS1                 ; Cell * _kIRS1Inact*IRS1pS636_639
        R3f     : PI3K => pPI3K                         ; Cell * _kPI3KPhos*PI3K*IRS1a
        R3b     : pPI3K => PI3K                         ; Cell * _kPI3KDephos*pPI3K
        R4f     : Akt => AktpT308                       ; Cell * MMWithKcat(_kAktPhos_km, _kAktPhos_kcat, Akt, pPI3K)
        R4b     : AktpT308 => Akt                       ; Cell * _kAktDephos*pPI3K*AktpT308
        R5f     : TSC2 => TSC2pT1462                    ; Cell * _kTSC2Phos*TSC2*AktpT308
        R5b     : TSC2pT1462 => TSC2                    ; Cell * _kTSC2Dephos*TSC2pT1462
        R6f     : mTORC1cyt => mTORC1lys                ; Cell * _kmTORC1cytToLys*mTORC1cyt*AA
        R6b     : mTORC1lys => mTORC1cyt                ; Cell * _kmTORC1LysToCyt*mTORC1lys
        R7f     : mTORC1lys => pmTORC1                  ; Cell * _kmTORC1Phos*mTORC1lys*TSC2
        R7b     : pmTORC1 => mTORC1lys                  ; Cell * _kmTORC1Dephos*pmTORC1
        R8f     : PRAS40 => PRAS40pT246                 ; Cell * _kPras40Phos*PRAS40*AktpT308
        R8b     : PRAS40pT246 => PRAS40                 ; Cell * _kPras40Dephos*PRAS40pT246
        R9f     : FourEBP1 => FourEBP1pT37_46           ; Cell * _kFourEBP1Phos*FourEBP1*pmTORC1
        R9b     : FourEBP1pT37_46 => FourEBP1           ; Cell * _kFourEBP1Dephos*FourEBP1pT37_46
        R10f    : S6K => S6KpT389                       ; Cell * _kS6KPhos*S6K*pmTORC1
        R10b    : S6KpT389 => S6K                       ; Cell * _kS6KDephos*S6KpT389

    end

    """