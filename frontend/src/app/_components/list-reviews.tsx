"use client";

import { Review } from "~/server/db/schema";
import { api } from "~/trpc/react";
import './list-reviews.css'

//** typescript prop pain */
// interface IProps {
//     reviews: Review[]
//   }
  
//** example movie selection dropdown */
export function ListReviews() {
  // const [movieNum, setMovieNum] = useState("2");

  // the useQuery hook allows api calls to contain state
  // check out https://tanstack.com/query/v5/docs/framework/react/guides/queries
  // for the full list of states
  const { data: latestReviews, isLoading: isGetting } =
  api.review.getXLatest.useQuery(20);
  // .useQuery(Number(movieNum))
  console.log("lr",latestReviews)
  const lReviews = latestReviews ?? [] 
  return (
    <div className="list-reviews">
      {lReviews.map((review)=><p>{review.movie.title} - {review.rating} stars &#9733;</p>)}
    </div>
  );
}
