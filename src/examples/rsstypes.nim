import options
import yottadb
when compileOption("profiler"):
  import std/nimprof
  
const
    MIN_KEYWORD_LEN* = 2
    MAX_SEARCH_RESULTS* = 1000
    MAX_CARDS* = 100
    DEFAULT_LANGUAGE* = "EN"

type
    Feed* = object of RootObj
        rssid*: string
        group*: string
        link*: string
        title*: string
        description*: string
        lastAccess*: string
        lastError*: string
        enabled*: bool = true

    ConfigFeed* = object of Feed
    
    UserFeeds* = object of RootObj
        userid*: string
        group*: string
        feeds*: seq[Feed]

    TimeSearchEntry* = object
        time*: int
        wordCount*: int
        subscript*: seq[string]

    SortBy* = enum 
        ByTodayDescending,
        ByTodayAscending,
        ByTimeDescending,
        ByTimeAscending,
        ByRelevanceDescending,
        ByRelevanceAscending

    SearchType* = enum 
        Basic,       # Clear, 0..100
        Incremental, # Prepend at 0
        Append       # Append at 100

    SearchParams* = object
        userid*: string
        keyword*: string
        lang*: string
        sort*: string
        direction*: string
        format*: string
        lastIdxRef*: string
        maxArticles*: int
        todayFrom*: int
        todayTo*: int
        lastPubdate*: int = int.high
        lowerBoundPubdate*: int
        sortBy*: SortBy
        searchType*: SearchType = Basic


    ListDirection* = enum
        Up,
        Down


    DBStats* = object
        ordercnt*: int
        querycnt*: int
        keylen*: int
        valuelen*: int
        minkey*: int = 999
        minvalue*: int = 999
        maxkey*: int
        maxvalue*: int
        global*: string
        duration*: string


    Registration* = object
        userid*: string
        name*: string
        password*: string
        email* {.INDEX: "userid".} : string
        message*: string
        country*: string
        plan*: string = "starter"
        status*: string
        time*: string
        terms*: bool


# Create the RSSATOM types.
type
  FeedType* = enum
    RSSv2,Atom

  RSS* = object
    id*: Option[string] #atom
    title*: Option[string]
    link*: Option[string]
    description*: Option[string]
    language*: Option[string]
    copyright*: Option[string]
    managingEditor*: Option[string]
    webMaster*: Option[string]
    pubDate*: Option[string]
    lastBuildDate*: Option[string]
    category*: seq[string]
    generator*: Option[string]
    docs*: Option[string]
    cloud*: RSSCloud
    ttl*: Option[string]
    image*: RSSImage
    rating*: Option[string]
    textInput*: RSSTextInput
    skipHours*: seq[string]
    skipDays*: seq[string]
    items*: seq[RSSItem]
    author*: Author
    feedType*: FeedType

  RSSEnclosure* = object
    url*: string
    length*: string
    enclosureType*: string

  RSSCloud* = object
    domain*: Option[string]
    port*: Option[string]
    path*: Option[string]
    registerProcedure*: Option[string]
    protocol*: Option[string]

  RSSImage* = object
    url*: Option[string]
    title*: Option[string]
    link*: Option[string]
    width*: Option[string]
    height*: Option[string]
    description*: Option[string]

  RSSTextInput* = object
    title*: Option[string]
    description*: Option[string]
    name*: Option[string]
    link*: Option[string]

  RSSItem* = object
    idxref* {.INDEX: "feedId".} : string
    title*: Option[string]
    link*: Option[string]
    description*: Option[string]
    content*: Option[string]
    author*: Author
    category*: seq[string]
    comments*: Option[string]
    enclosure*: RSSEnclosure
    guid* {.INDEX: "idxref".} : Option[string]
    pubDate* {.INDEX: "idxref".} : Option[string]
    sourceUrl*: Option[string]
    sourceText*: Option[string]
    updated*: Option[string] #Atom
    feedId*: Option[string]
    topic*: Option[string]
    keywords*: seq[string]

  Author* = object
    name*: Option[string]
    email*: Option[string]
    uri*: Option[string]
    #legacyRss*: Option[string] #RSSV2

