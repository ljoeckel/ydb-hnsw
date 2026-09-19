import yottadb

proc backup(gbl: string) =
    let dest = gbl & "BACKUP"
    for (k,v) in QueryItr @gbl:
        Set: @dest(k) = v


let globals = @[
    "^DBSTATS", "^DBStats", "^DBStatsDOMAIN", "^DBStatsDetail", "^Feed",
    "^HNSWArticlesKEY", "^HNSWArticlesMETA", "^HNSWArticlesNODE", "^HNSWArticlesQKEY",
    "^HNSWArticlesQMETA", "^HNSWArticlesQNODE", "^RSS", "^RSSArchive",
    "^RSSCNT", "^RSSEnclosure", "^RSSFTI", "^RSSImage", "^RSSItem", "^RSSItemFTI",
    "^RSSItemGUID", "^RSSItemIDXREF", "^RSSItemPUBDATE", "^Session", "^UserFeeds",
    "^stopwordsALL",  "^stopwordsDE", "^stopwordsEN", "^stopwordsES", "^stopwordsWC",
    ]