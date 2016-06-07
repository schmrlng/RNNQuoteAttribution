using StatsBase

speaker_counts = countmap(ASCIIString[])
word_counts = countmap(ASCIIString[])
open("prideprejudice.txt", "w") do writefile
    open("PridePrejudice_Austen/REAL_ALL_CONTENTS_PP.txt") do f
        lines = readlines(f)
        chapters = [split(l, '\t')[1] for l in lines]
        speakers = [lowercase(strip(ascii(split(l, '\t')[2]))) for l in lines]
        quotes = [split(l, '\t')[3] for l in lines]
        addcounts!(speaker_counts, speakers)


        BATCHSIZE = 200
        N  = length(quotes)
        tokenized = SubString{ASCIIString}[]
        for rg in (i:min(i+BATCHSIZE-1, N) for i in 1:BATCHSIZE:N)
            @show rg
            allquotes = join(quotes[rg], " delimdelimdelim ")
            append!(tokenized, map(strip, split(
                readstring(pipeline(`echo $allquotes`, `java edu.stanford.nlp.process.PTBTokenizer -lowerCase -options ptb3Escaping=false`))
                , "delimdelimdelim")))
        end

        for q in tokenized
            if q == ""
                warn("empty quote in $fname")
            else
                addcounts!(word_counts, map(ascii, split(q)))
            end
        end
        @assert length(chapters) == length(speakers) == length(tokenized)
        for (c,s,q) in zip(chapters, speakers, tokenized)
            write(writefile, c * "\t" * s * "\t" * strip(replace(q, "\n", " ")) * "\n")
        end
    end
end

(sort(collect(speaker_counts), by=last), sort(collect(word_counts), by=last))